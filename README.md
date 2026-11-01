# Toolbox

General tools for Maarten's Claude chats. Public on purpose: GitHub Actions minutes are free for public repos
(private repos share the 2,000 free minutes a month that the day-trading bot needs). Nothing secret goes in here.

## Video analysis
Paste TikTok / YouTube Shorts / Instagram / X links (single videos or whole accounts) in a Claude chat that has this repo attached
("attach Mjeerten/Toolbox"). Claude runs `.github/workflows/video.yml`: it downloads each video, saves a frame every
second, transcribes the speech and stores caption, account, date and views on the `videos` branch.

Account links (e.g. https://www.tiktok.com/@name) list all the account's videos and process the newest 10 (input `max`).

Without Claude: Actions -> video -> Run workflow -> paste the link(s), separated by spaces.
