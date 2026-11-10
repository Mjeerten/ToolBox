# Toolbox — notes for Claude

Shared tools for Maarten (Amsterdam). Public repo so Actions minutes are free; never put keys or personal data here.
Related repos: day-trading-desk (Nasdaq, private), Meme-Desk (memecoin research, public).

- Crazy strategies (day desk round 33b): `research/crazy.py` + `.github/workflows/crazy.yml` (push or dispatch): random
  rules on ~70 markets, evolved on 2019-23, holdout 2024+; report in `results/crazy/crazy_report.txt`. 1-minute data via
  `tools/fetch_histdata.sh` (HistData) and data.binance.vision.
- Video and account links (TikTok etc.): `.github/workflows/video.yml` runs `tools/video.py`. Inputs `urls`
  (videos and/or accounts, space-separated), `every` (seconds per frame, default 1), `max` (newest videos per account,
  default 10). Output on the `videos` branch:
  - per video `videos/<extractor>-<id>/`: info.txt (caption, views, date), transcript.txt (speech), ocr.txt (on-screen
    text with the second it appears, every frame), sheets/NN.jpg (12 frames each, second in the corner; only frames
    where the picture changes, at least one every 4 s, so read all sheets), frames/NNNN.jpg (every frame).
  - per slideshow (TikTok photo post, /photo/ link; yt-dlp can't): same folder `videos/TikTok-<id>/`, frames/NNNN.jpg =
    slide N full size, sheets of every slide, ocr.txt per slide, transcript.txt of the sound. Slides come from the
    post page's JSON (`slideshow` in video.py), fallback tikwm.com API; vm.tiktok.com short links are resolved first.
  - per account `videos/account-<extractor>-<name>/index.txt`: every listed video (date, views, likes, caption, link)
    and the folders of the processed ones.
  Trigger: `gh api -X POST repos/Mjeerten/toolbox/actions/workflows/video.yml/dispatches -f ref=main -f 'inputs[urls]=...'`
  (add `-f 'inputs[max]=N'`). Read: `git fetch origin videos` + `git show FETCH_HEAD:<path>` (artifact downloads and
  TikTok are blocked from Claude's container, so everything runs on GitHub).
- TikTok 'Video not available, status code 10231' = TikTok hides that video from logged-out viewers (old links still
  work, so the tool is fine). Two routes: (1) Maarten saves the video (Share > Save video) and uploads it to `inbox/` on
  main (github.com: Add file > Upload files): the push runs video.yml on each new inbox file -> `videos/upload-<name>/`.
  (2) Secret TIKTOK_COOKIES = cookies.txt export of his own TikTok login (browser extension 'Get cookies.txt LOCALLY',
  tiktok.com tab); yt-dlp then sees what his account sees. Best a spare TikTok account; renew when it stops working.
- ALWAYS LOOK AT THE PICTURES, every video, not only the transcript: read every sheet (sheets/*.jpg) with the Read
  tool, plus ocr.txt; open single frames (frames/NNNN.jpg, crop/enlarge with ffmpeg) wherever there are numbers,
  settings, indicator panels, P&L, wallet/contract addresses or chart levels. Music-only clips have lyric transcripts;
  the frames are the content. Report what the screens show (and whether they contradict the caption/voice-over).
- Indicator accounts: write down each indicator's name, inputs/settings and buy/sell rules from the frames, and where
  it can be found (TradingView name, link in caption). Strategies or indicators with clear rules go to the day desk as
  a backtest round (judge 2019-23, holdout 2024+). Earlier analyses: day desk BACKTEST.md rounds 18-20, Meme-Desk
  RESULTS.md and its `videos` branch.
