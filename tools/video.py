"""Turn video links and account links into things Claude can read: frames, contact sheets, on-screen text (OCR),
a speech transcript and caption / account info. Used by .github/workflows/video.yml.

  python tools/video.py --out videos --every 1 --max 10 URL [URL ...]

Per video  -> <out>/<extractor>-<id>/ info.txt, transcript.txt, ocr.txt, sheets/NN.jpg (12 frames each, 4x3,
              timestamps burned in), frames/NNNN.jpg (frame k = second (k-1)*every).
              Sheets hold only the frames where the picture changes (plus at least one every MAX_GAP seconds), so a
              talking head does not fill ten sheets; every frame stays in frames/ and is OCR'd (4 at a time).
Slideshow  (TikTok photo post, /photo/ link) -> the same folder layout: frames/NNNN.jpg = slide N, sheets of every
              slide ("slide N" in the corner), ocr.txt per slide, transcript.txt of the background sound, info.txt.
              yt-dlp has no photo support, so the slides come from the post's page (its embedded JSON), with
              tikwm.com as a fallback; all slides download in parallel.
Per account (TikTok @user, YouTube channel, Instagram profile...) -> <out>/account-<extractor>-<name>/index.txt
              (every listed video: date, views, likes, duration, caption, link) and the newest `max` videos processed
              as above (their folders are listed at the end of index.txt).
"""
import argparse
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OCR_ENV = dict(os.environ, OMP_THREAD_LIMIT="1")
DIFF, MAX_GAP = 6.0, 4   # sheet frame kept when its 48x48 grey thumbnail differs this much (0-255 mean) / every 4 s

YTDLP = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings"]
COOKIES = os.environ.get("COOKIES_FILE") if os.path.getsize(os.environ.get("COOKIES_FILE") or os.devnull) else None


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")


def fetch(url, data=None, timeout=30):
    """(final url, bytes) with a browser user agent; follows redirects (vm.tiktok.com short links)."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Referer": "https://www.tiktok.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl(), r.read()


def resolve(url):
    """Short TikTok links (vm.tiktok.com/..., tiktok.com/t/...) -> the full link, so a /photo/ post is recognised."""
    if os.path.isfile(url) or not re.search(r"//(vm|vt)\.tiktok\.com/|tiktok\.com/t/", url):
        return url
    try:
        return fetch(url)[0]
    except Exception as e:   # noqa: keep the short link, yt-dlp resolves it itself
        print("resolve failed:", url, repr(e)[:200])
        return url


def sh(cmd, text=True):
    return subprocess.run(cmd, capture_output=True, text=text)


def probe(url, max_n):
    if os.path.isfile(url):
        return None
    r = sh(YTDLP + (["--cookies", COOKIES] if COOKIES else []) + ["--flat-playlist", "--playlist-end", str(max(max_n * 3, 30)), "-J", url])
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
        # one thread per tesseract (OMP_THREAD_LIMIT=1): 4 copies with their own thread pools on 4 CPUs crawl
        texts = list(ex.map(lambda f: subprocess.run(["tesseract", f, "-", "--psm", "11"], capture_output=True,
                                                     text=True, timeout=60, env=OCR_ENV).stdout, frames))
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


def sheets(frames, keep, every, folder, name_of=None):
    """Contact sheets of the kept frames: 4x3, 360 px wide each, the second (or name_of(k)) in the corner."""
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
            label = name_of(k) if name_of else f"{k * every:g}s"
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


def download(url, tmp):
    """yt-dlp (with Maarten's own TikTok login when the TIKTOK_COOKIES secret is set: videos TikTok only shows to
    logged-in / adult viewers answer 'status 10231' without it), or a file uploaded to inbox/. Returns the files."""
    if os.path.isfile(url):                      # uploaded file: inbox/<name>.mp4
        name = re.sub(r"[^\w.-]", "_", os.path.splitext(os.path.basename(url))[0])
        vid = f"{tmp}/upload-{name}{os.path.splitext(url)[1] or '.mp4'}"
        shutil.copy(url, vid)
        json.dump({"uploader": "(uploaded file)", "description": f"uploaded file {os.path.basename(url)}"},
                  open(f"{tmp}/upload-{name}.info.json", "w"))
        return [vid]
    cookies = ["--cookies", COOKIES] if COOKIES else []
    r = sh(YTDLP + cookies + ["--no-playlist", "-f", "b[height<=1080]/b", "--write-info-json",
                              "-o", f"{tmp}/%(extractor)s-%(id)s.%(ext)s", url])
    vids = [f for f in glob.glob(f"{tmp}/*") if not f.endswith(".info.json")]
    if r.returncode == 0 and vids:
        return vids
    hint = (" (TikTok hides this video from logged-out viewers: add the TIKTOK_COOKIES secret, see CLAUDE.md, "
            "or upload the saved video to inbox/)") if "10231" in r.stderr and not COOKIES else ""
    print("DOWNLOAD FAILED:", url, r.stderr[-300:] + hint)
    return None


def find_item(obj):
    """The post dict inside TikTok's page JSON: the first dict with an 'imagePost' key (its place moves between
    page versions, so search instead of a fixed path)."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if isinstance(o.get("imagePost"), dict):
                return o
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return None


def best_url(urls):
    """A slide URL Pillow can open: prefer jpeg / webp over heic."""
    urls = [u for u in urls or [] if u]
    return next((u for u in urls if re.search(r"jpe?g|webp", u, re.I)), urls[0] if urls else None)


def slides_from_page(url):
    """Slides + post info from the photo post's own page."""
    page = fetch(url)[1].decode("utf-8", "replace")
    m = re.search(r'<script[^>]+id="(?:__UNIVERSAL_DATA_FOR_REHYDRATION__|SIGI_STATE)"[^>]*>(.*?)</script>', page, re.S)
    item = find_item(json.loads(m.group(1))) if m else None
    if not item:
        raise ValueError("no imagePost in the page JSON")
    st, au, mu = item.get("stats") or {}, item.get("author") or {}, item.get("music") or {}
    return dict(id=item.get("id"), images=[best_url((im.get("imageURL") or {}).get("urlList"))
                                           for im in item["imagePost"].get("images") or []],
                desc=item.get("desc") or (item["imagePost"].get("title") or ""), uploader=au.get("uniqueId"),
                uploader_id=au.get("id"), date=item.get("createTime"), views=st.get("playCount"),
                likes=st.get("diggCount"), comments=st.get("commentCount"), music=mu.get("playUrl"))


def slides_from_tikwm(url):
    """Fallback: the public tikwm.com API (only the post link is sent)."""
    raw = fetch("https://www.tikwm.com/api/", data=urllib.parse.urlencode({"url": url, "hd": 1}).encode())[1]
    d = json.loads(raw).get("data") or {}
    if not d.get("images"):
        raise ValueError(f"tikwm: no images ({raw[:200]!r})")
    return dict(id=d.get("id"), images=d["images"], desc=d.get("title") or "",
                uploader=(d.get("author") or {}).get("unique_id"), uploader_id=(d.get("author") or {}).get("id"),
                date=d.get("create_time"), views=d.get("play_count"), likes=d.get("digg_count"),
                comments=d.get("comment_count"), music=d.get("music"))


def slideshow(url, out, whisper):
    """A TikTok photo post: every slide as a frame, sheets, OCR per slide, the sound's transcript, info.txt."""
    post, errors = None, []
    for get in (slides_from_page, slides_from_tikwm):
        try:
            post = get(url)
            break
        except Exception as e:   # noqa: try the next source
            errors.append(f"{get.__name__}: {e!r}"[:300])
    if not post:
        print("SLIDESHOW FAILED:", url, " / ".join(errors))
        return None
    pid = post["id"] or re.search(r"/photo/(\d+)", url).group(1)
    folder = f"{out}/TikTok-{pid}"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(f"{folder}/frames")
    os.makedirs(f"{folder}/sheets")

    def save(job):
        n, u = job
        try:
            im = Image.open(io.BytesIO(fetch(u, timeout=60)[1])).convert("RGB")
        except Exception as e:   # noqa: one bad slide must not lose the rest
            print(f"slide {n} failed: {e!r}"[:200])
            return None
        if im.width > 1080:
            im = im.resize((1080, round(im.height * 1080 / im.width)))
        path = f"{folder}/frames/{n:04d}.jpg"
        im.save(path, quality=88)
        return path

    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(save, [(n, u) for n, u in enumerate(post["images"], 1) if u]))
    frames = [f for f in got if f]
    print(f"-- {folder}: {len(frames)} of {len(post['images'])} slides, OCR + transcript...", flush=True)
    nums = [int(os.path.basename(f)[:4]) for f in frames]
    sheets(frames, list(range(len(frames))), 1, folder, name_of=lambda k: f"slide {nums[k]}")
    with ThreadPoolExecutor(4) as ex:   # every slide's full text (slides are separate pages: no line dedup)
        texts = list(ex.map(lambda f: subprocess.run(["tesseract", f, "-", "--psm", "3"], capture_output=True,
                                                     text=True, timeout=120, env=OCR_ENV).stdout, frames))
    ocr = []
    for n, t in zip(nums, texts):
        lines = [re.sub(r"\s+", " ", x).strip() for x in t.splitlines()]
        ocr += [f"--- slide {n}"] + ([x for x in lines if x] or ["(no text found)"])
    open(f"{folder}/ocr.txt", "w").write("\n".join(ocr or ["(no slides)"]) + "\n")
    tr = ["(no sound link)"]
    if post.get("music") and whisper:
        try:
            snd = "/tmp/dl/sound.mp3"
            os.makedirs("/tmp/dl", exist_ok=True)
            open(snd, "wb").write(fetch(post["music"], timeout=60)[1])
            tr = transcript(snd, whisper)
        except Exception as e:   # noqa: a transcript is optional
            tr = [f"(transcript failed: {e!r})"]
    open(f"{folder}/transcript.txt", "w").write("\n".join(tr) + "\n")
    info = [f"url: {url}", f"uploader: {post['uploader']} ({post['uploader_id']})", f"date: {fmt_date(post['date'])}",
            "type: slideshow (photo post)",
            f"views: {post['views']}  likes: {post['likes']}  comments: {post['comments']}",
            f"slides: {len(frames)} of {len(post['images'])} (frames/NNNN.jpg = slide N, full size; all on the sheets; "
            "ocr.txt has each slide's text)", "", "title/caption:", post["desc"].strip()]
    open(f"{folder}/info.txt", "w").write("\n".join(info) + "\n")
    print(f"== {folder}\n" + "\n".join(info) + "\n" + "\n".join(ocr[:80]), flush=True)
    return folder


def video(url, out, every, whisper):
    url = resolve(url)
    if "/photo/" in url:
        return slideshow(url, out, whisper)
    tmp = "/tmp/dl"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    vids = download(url, tmp)
    if not vids:
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
    print(f"-- {folder}: {len(frames)} frames, OCR + transcript...", flush=True)
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
        if os.path.isfile(url):
            name = re.sub(r"[^\w.-]", "_", os.path.splitext(os.path.basename(url))[0])
            if os.path.isdir(f"{a.out}/upload-{name}"):
                print("already done:", url)
                continue
        url = resolve(url)
        if "/photo/" in url:
            slideshow(url, a.out, whisper)
            continue
        info = probe(url, a.max)
        if info and info.get("_type") == "playlist":
            account(info, url, a.out, a.max, every, whisper)
        else:
            video(url, a.out, every, whisper)


if __name__ == "__main__":
    main()
