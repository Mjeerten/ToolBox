# Toolbox — notes for Claude

Shared tools for Maarten (Amsterdam). Public repo so Actions minutes are free; never put keys or personal data here.
Related repos: day-trading-desk (Nasdaq, private), Meme-Desk (memecoin research, public).

- Video and account links (TikTok etc.): `.github/workflows/video.yml` runs `tools/video.py`. Inputs `urls`
  (videos and/or accounts, space-separated), `every` (seconds per frame, default 1), `max` (newest videos per account,
  default 10). Output on the `videos` branch:
  - per video `videos/<extractor>-<id>/`: info.txt (caption, views, date), transcript.txt (speech), ocr.txt (on-screen
    text with the second it appears), sheets/NN.jpg (12 frames each, second in the corner), frames/NNNN.jpg.
  - per account `videos/account-<extractor>-<name>/index.txt`: every listed video (date, views, likes, caption, link)
    and the folders of the processed ones.
  Trigger: `gh api -X POST repos/Mjeerten/toolbox/actions/workflows/video.yml/dispatches -f ref=main -f 'inputs[urls]=...'`
  (add `-f 'inputs[max]=N'`). Read: `git fetch origin videos` + `git show FETCH_HEAD:<path>` (artifact downloads and
  TikTok are blocked from Claude's container, so everything runs on GitHub).
- ALWAYS LOOK AT THE PICTURES, every video, not only the transcript: read every sheet (sheets/*.jpg) with the Read
  tool, plus ocr.txt; open single frames (frames/NNNN.jpg, crop/enlarge with ffmpeg) wherever there are numbers,
  settings, indicator panels, P&L, wallet/contract addresses or chart levels. Music-only clips have lyric transcripts;
  the frames are the content. Report what the screens show (and whether they contradict the caption/voice-over).
- Indicator accounts: write down each indicator's name, inputs/settings and buy/sell rules from the frames, and where
  it can be found (TradingView name, link in caption). Strategies or indicators with clear rules go to the day desk as
  a backtest round (judge 2019-23, holdout 2024+). Earlier analyses: day desk BACKTEST.md rounds 18-20, Meme-Desk
  RESULTS.md and its `videos` branch.
