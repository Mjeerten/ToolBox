"""Turn video links and account links into things Claude can read: frames, contact sheets, on-screen text (OCR),
a speech transcript and caption / account info. Used by .github/workflows/video.yml.

  python tools/video.py --out videos --every 1 --max 10 URL [URL ...]

Per video  -> <out>/<extractor>-<id>/ info.txt, transcript.txt, ocr.txt, sheets/NN.jpg (12 frames each, 4x3,
              timestamps burned in), frames/NNNN.jpg (frame k = second (k-1)*every).
              Sheets hold only the frames where the picture changes (plus at least one every MAX_GAP seconds), so a
              talking head does not fill ten sheets; every frame stays in frames/ and is OCR'd (4 at a time).
Per account (TikTok @user, YouTube channel, Instagram profile...) -> <out>/account-<extractor>-<name>/index.txt
              (every listed video: date, views, likes, duration, caption, link) and the newest `max` videos processed
              as above (their folders are listed at the end of index.txt).
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIFF, MAX_GAP = 6.0, 4   # sheet frame kept when its 48x48 grey thumbnail differs this much (0-255 mean) / every 4 s

YTDLP = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings"]


def sh(cmd, text=True):
    return subprocess.run(cmd, capture_output=True, text=text)


def probe(url, max_n):
    r = sh(YTDLP + ["--flat-playlist", "--playlist-end", str(max(max_n * 3, 30)), "-J", url])
    if r.returncode != 0:
        print("PROBE FAILED:", url, r.stderr[-400:])
        return None
    return json.loads(r.stdout)


def fmt_date(d):
    d = str(d or "")
    if len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    try:
        return datetime.datetime.utcfromtimestamp(float(d)).strftime("%Y-%m-%d")
    except ValueError:
        return d


def account(info, url, out, max_n, every, whisper):
    m = re.search(r"/@([\w.-]+)", url)
    name = re.sub(r"[^\w.-]", "_", m.group(1) if m else str(info.get("uploader") or info.get("title") or info.get("id")))
    folder = f"{out}/account-{info.get('extractor_key', 'web')}-{name}"
    os.makedirs(folder, exist_ok=True)
    entries = [e for e in info.get("entries") or [] if e]
    lines = [f"account: {info.get('uploader') or info.get('title')} ({name})", f"url: {url}",
             f"followers: {info.get('channel_follower_count')}", f"listed videos: {len(entries)} (newest first)",
             "", "date | views | likes | duration_s | caption | link"]
    for e in entries:
        cap = (e.get("description") or e.get("title") or "").replace("\n", " ")[:200]
        link = e.get("webpage_url") or e.get("url")
        lines.append(f"{fmt_date(e.get('upload_date') or e.get('timestamp'))} | {e.get('view_count')} | "
                     f"{e.get('like_count')} | {e.get('duration')} | {cap} | {link}")
    done = []
    for e in entries[:max_n]:
        link = e.get("webpage_url") or e.get("url")
        if link:
            d = video(link, out, every, whisper)
            if d:
                done.append(d)
    lines += ["", "processed (newest first):"] + done
    open(f"{folder}/index.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


def ocr_frames(frames, every):
    """Text per frame with tesseract; only lines that change are kept, with the second they appear."""
    out, last = [], set()
    with ThreadPoolExecutor(4) as ex:
        texts = list(ex.map(lambda f: sh(["tesseract", f, "-", "--psm", "11"]).stdout, frames))
    for k, raw in enumerate(texts):
        txt = {re.sub(r"\s+", " ", t).strip() for t in raw.splitlines()}
        txt = {t for t in txt if len(t) >= 3 and sum(ch.isalnum() for ch in t) >= 3}
        new = sorted(txt - last)
        if new:
            out.append(f"[{k * every:5.0f}s] " + " | ".join(new))
        last = txt
    return out


def key_frames(frames, every):
    """Indexes of frames that look different from the last kept one, or MAX_GAP seconds after it."""
    keep, last, last_t = [], None, -1e9
    for k, f in enumerate(frames):
        g = np.asarray(Image.open(f).convert("L").resize((48, 48)), np.float32)
        if last is None or np.abs(g - last).mean() >= DIFF or k * every - last_t >= MAX_GAP:
            keep.append(k)
            last, last_t = g, k * every
    return keep


def sheets(frames, keep, every, folder):
    """Contact sheets of the kept frames: 4x3, 360 px wide each, the second in the corner."""
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    for n in range(0, len(keep), 12):
        tiles = []
        for k in keep[n:n + 12]:
            im = Image.open(frames[k]).convert("RGB")
            im = im.resize((360, round(im.height * 360 / im.width)))
            d = ImageDraw.Draw(im)
            label = f"{k * every:g}s"
            d.rectangle((0, 0, 12 + 22 * len(label), 44), fill=(0, 0, 0))
            d.text((6, 4), label, fill=(255, 255, 0), font=font)
            tiles.append(im)
        w, h = 360, max(t.height for t in tiles)
        sheet = Image.new("RGB", (w * 4, h * ((len(tiles) + 3) // 4)), (0, 0, 0))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * w, (i // 4) * h))
        sheet.save(f"{folder}/sheets/{n // 12 + 1:02d}.jpg", quality=80)


def transcript(path, whisper):
    raw = sh(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
             text=False).stdout
    if not raw:
        return ["(no audio)"]
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    segs, info = whisper.transcribe(audio, vad_filter=True)
    lines = [f"language: {info.language} ({info.language_probability:.0%})"]
    lines += [f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}" for s in segs]
    if len(lines) == 1:
        lines.append("(no speech found: music only? read the frames / ocr.txt)")
    return lines


def video(url, out, every, whisper):
    tmp = "/tmp/dl"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    r = sh(YTDLP + ["--no-playlist", "-f", "b[height<=1080]/b", "--write-info-json",
                    "-o", f"{tmp}/%(extractor)s-%(id)s.%(ext)s", url])
    vids = [f for f in glob.glob(f"{tmp}/*") if not f.endswith(".info.json")]
    if r.returncode != 0 or not vids:
        print("DOWNLOAD FAILED:", url, r.stderr[-300:])
        return None
    vid = vids[0]
    vid_id = os.path.splitext(os.path.basename(vid))[0]
    folder = f"{out}/{vid_id}"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(f"{folder}/frames")
    os.makedirs(f"{folder}/sheets")
    sh(["ffmpeg", "-loglevel", "error", "-i", vid, "-vf", f"fps=1/{every},scale=720:-2", "-q:v", "4",
        f"{folder}/frames/%04d.jpg"])
    frames = sorted(glob.glob(f"{folder}/frames/*.jpg"))
    keep = key_frames(frames, every) if frames else []
    sheets(frames, keep, every, folder)
    i = json.load(open(glob.glob(f"{tmp}/*.info.json")[0]))
    info = [f"url: {url}", f"uploader: {i.get('uploader') or i.get('channel')} ({i.get('uploader_id')})",
            f"date: {fmt_date(i.get('upload_date'))}", f"duration_s: {i.get('duration')}",
            f"views: {i.get('view_count')}  likes: {i.get('like_count')}  comments: {i.get('comment_count')}",
            f"frames: {len(frames)}, one every {every} s (frames/0001.jpg = second 0); "
            f"sheets: {len(glob.glob(folder + '/sheets/*.jpg'))} (12 frames each; {len(keep)} frames where the picture "
            f"changes, at least one every {MAX_GAP} s; all frames in frames/)",
            "", "title/caption:", (i.get("description") or i.get("title") or "").strip()]
    open(f"{folder}/info.txt", "w").write("\n".join(info) + "\n")
    open(f"{folder}/ocr.txt", "w").write("\n".join(ocr_frames(frames, every) or ["(no text found)"]) + "\n")
    try:
        tr = transcript(vid, whisper)
    except Exception as e:   # noqa: a transcript is optional
        tr = [f"(transcript failed: {e!r})"]
    open(f"{folder}/transcript.txt", "w").write("\n".join(tr) + "\n")
    print(f"== {folder}\n" + "\n".join(info) + "\n" + "\n".join(tr[:60]), flush=True)
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--every", type=float, default=1)
    ap.add_argument("--max", type=int, default=10)
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    os.makedirs(a.out, exist_ok=True)
    every = int(a.every) if float(a.every).is_integer() else a.every
    for url in a.urls:
        info = probe(url, a.max)
        if info and info.get("_type") == "playlist":
            account(info, url, a.out, a.max, every, whisper)
        else:
            video(url, a.out, every, whisper)


if __name__ == "__main__":
    main()
