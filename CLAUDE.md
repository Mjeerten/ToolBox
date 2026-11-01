# Toolbox — notes for Claude

Shared tools for Maarten (Amsterdam). Public repo so Actions minutes are free; never put keys or personal data here.
Related repos: day-trading-desk (Nasdaq, private), Meme-Desk (memecoin research, public).

- Video links (TikTok etc.): `.github/workflows/video.yml`, inputs `urls` (space-separated) and `every` (seconds per
  frame, default 1). The runner downloads with yt-dlp, saves frames (720 px wide), a faster-whisper transcript and
  caption info, and commits them to the `videos` branch as `videos/<extractor>-<id>/` (info.txt, transcript.txt,
  frames/0001.jpg = second 0). Main stays clean.
  Trigger: `gh api -X POST repos/Mjeerten/toolbox/actions/workflows/video.yml/dispatches -f ref=main -f 'inputs[urls]=...'`.
  Read: `git fetch origin videos` + `git show FETCH_HEAD:<path>` (artifact downloads are blocked from Claude's
  container; TikTok too, so everything runs on GitHub). A contact sheet helps: ffmpeg `tile=5x2` on the frames.
  Music-only clips give lyric transcripts; read the on-screen text in the frames. Strategies with clear rules go to
  the day desk as a backtest round (judge 2019-23, holdout 2024+). Earlier analyses: Meme-Desk RESULTS.md, `videos` branch.
