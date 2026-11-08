"""Copy-trading test (8 Nov 2026): does copying famous funds (13F filings) or US politicians (House trade reports)
beat just holding SPY / QQQ, when you can only copy from the day the filing is public?

Runs on GitHub Actions in this public repo (.github/workflows/copytrade.yml, free minutes); Claude's container cannot
reach SEC, House or Yahoo. Results go to swing-desk BACKTEST.md.
  python -m research.copytrade etf     real ETFs that do this: GURU (hedge-fund 13F top picks, since 2012),
                                       NANC / KRUZ (Congress Democrats / Republicans, since Feb 2023)
  python -m research.copytrade f13     own test: top 10 holdings of famous funds, bought the day after each 13F filing
  python -m research.copytrade house-fetch YEAR   download + read one year of House reports (one job per year)
  python -m research.copytrade house   own test: every House member's stock purchase, bought the day after it is filed
Judge 2019-2023, holdout 2024+ (2014-2018 shown as extra). Costs: 0.1% per trade side.
"""
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# SEC answers 403 unless the User-Agent names a contact ("Name email"): secret SEC_UA, set by Maarten
UA = {"User-Agent": os.environ.get("SEC_UA") or "copy-trading research (github.com/Mjeerten/toolbox)",
      "Accept-Encoding": "gzip, deflate"}
CACHE = os.environ.get("COPY_CACHE", ".copycache")
PERIODS = [("2014-18", "2014-01-01", "2018-12-31"), ("2019-23", "2019-01-01", "2023-12-31"),
           ("2024+", "2024-01-01", "2100-01-01")]
COST = 0.001
FUNDS = {  # CIK: short name (names are checked against SEC's own name in the log)
    1067983: "Berkshire (Buffett)", 1336528: "Pershing Square (Ackman)", 1656456: "Appaloosa (Tepper)",
    1649339: "Scion (Burry)", 1061768: "Baupost (Klarman)", 1079114: "Greenlight (Einhorn)",
    1040273: "Third Point (Loeb)", 1167483: "Tiger Global", 1061165: "Lone Pine", 1103804: "Viking Global",
    1536411: "Duquesne (Druckenmiller)", 1709323: "Himalaya (Li Lu)", 1112520: "Akre", 1135730: "Coatue",
    1541617: "Altimeter", 1569205: "Fundsmith (Smith)", 1549575: "Pabrai", 1096343: "Markel",
}


def get(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None


# ---------------------------------------------------------------- prices (Yahoo, dividends included)
def prices(tk):
    os.makedirs(f"{CACHE}/px", exist_ok=True)
    f = f"{CACHE}/px/{tk}.pkl"
    if os.path.exists(f):
        return pd.read_pickle(f)
    s = None
    r = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1=1356998400&period2={int(time.time())}"
            "&interval=1d&events=div,split")
    try:
        j = r.json()["chart"]["result"][0]
        adj = j["indicators"]["adjclose"][0]["adjclose"]
        s = pd.Series(adj, index=pd.to_datetime(j["timestamp"], unit="s").normalize(), dtype=float).dropna()
        s = s[~s.index.duplicated(keep="last")]
        if len(s) < 5:
            s = None
    except Exception:  # noqa: unknown / delisted ticker
        s = None
    pd.to_pickle(s, f)
    time.sleep(0.3)
    return s


def stats(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index <= b)].dropna()
    if len(e) < 60:
        return None
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return cagr * 100, dd * 100


def table(curves, title):
    """curves: {name: equity series}. Prints %/yr and worst drop per period plus the excess over SPY / QQQ."""
    bench = {k: prices(k) for k in ("SPY", "QQQ")}
    print(f"\n=== {title}")
    print(f"{'':34s}" + "".join(f"{p:>26s}" for p, _, _ in PERIODS))
    print(f"{'':34s}" + "".join(f"{'%/yr  drop  vsSPY vsQQQ':>26s}" for _ in PERIODS))
    for name, eq in list(curves.items()) + [("SPY (hold)", bench["SPY"]), ("QQQ (hold)", bench["QQQ"])]:
        row = f"{name[:34]:34s}"
        if eq is None or eq.dropna().empty:
            print(row + "  no price data")
            continue
        for _, a, b in PERIODS:
            s = stats(eq, a, b)
            if s is None:
                row += f"{'-':>26s}"
                continue
            e = eq.dropna()
            a2, b2 = max(pd.Timestamp(a), e.index[0]), min(pd.Timestamp(b), e.index[-1])  # same dates for the index
            x = [stats(bench[k], a2, b2) for k in ("SPY", "QQQ")]
            row += f"{s[0]:7.1f}{s[1]:6.0f}{s[0] - x[0][0]:+7.1f}{s[0] - x[1][0]:+6.1f}"
        print(row)


# ---------------------------------------------------------------- portfolios from dated target weights
def run_book(targets, holds_until=None):
    """targets: list of (date, {ticker: weight}) rebalances, each held until the next one. Weights drift with the
    prices in between; tickers without prices are dropped and the rest scaled up (coverage is printed).
    Returns the equity curve (1.0 at the first rebalance)."""
    targets = sorted(targets, key=lambda x: x[0])
    idx = prices("SPY").index
    eq, val, cov = [], 1.0, []
    prev = {}
    for k, (d, w) in enumerate(targets):
        d = idx[idx.searchsorted(pd.Timestamp(d) + pd.Timedelta(days=1))] if pd.Timestamp(d) < idx[-1] else None
        if d is None:
            break
        end = pd.Timestamp(targets[k + 1][0]) + pd.Timedelta(days=1) if k + 1 < len(targets) else idx[-1]
        days = idx[(idx >= d) & (idx <= end)]
        if len(days) < 2:
            continue
        px = {t: prices(t) for t in w}
        ok = {t: x for t, x in w.items() if px[t] is not None and px[t].index[0] <= d and px[t].index[-1] >= days[-1]}
        tot = sum(ok.values())
        cov.append(tot / max(sum(w.values()), 1e-12))
        if not ok:
            continue
        ok = {t: x / tot for t, x in ok.items()}
        turn = sum(abs(ok.get(t, 0) - prev.get(t, 0)) for t in set(ok) | set(prev))
        val *= 1 - COST * turn
        rel = sum(x * px[t].reindex(days).ffill() / px[t].reindex(days).ffill().iloc[0] for t, x in ok.items())
        seg = val * rel
        eq.append(seg.iloc[:-1] if k + 1 < len(targets) else seg)
        val = float(seg.iloc[-1])
        last = rel.iloc[-1]
        prev = {t: x * float(px[t].reindex(days).ffill().iloc[-1] / px[t].reindex(days).ffill().iloc[0]) / last
                for t, x in ok.items()}
    if not eq:
        return None, 0
    e = pd.concat(eq)
    return e[~e.index.duplicated(keep="last")], float(np.mean(cov))


# ---------------------------------------------------------------- 1. real ETFs
def etf():
    curves = {f"{k} ({v})": prices(k) for k, v in
              [("GURU", "hedge-fund 13F picks"), ("NANC", "Congress Democrats"), ("KRUZ", "Congress Republicans")]}
    table(curves, "Real ETFs that copy funds / Congress (after their fees)")


# ---------------------------------------------------------------- 2. 13F filings
def cusip_map(cusips):
    """CUSIP -> US ticker via OpenFIGI (free, no key: 10 per request, 25 requests a minute)."""
    f = f"{CACHE}/figi.json"
    m = json.load(open(f)) if os.path.exists(f) else {}
    todo = [c for c in sorted(set(cusips)) if c not in m]
    print(f"OpenFIGI: {len(todo)} new CUSIPs", flush=True)
    for i in range(0, len(todo), 10):
        part = todo[i:i + 10]
        for k in range(5):
            r = requests.post("https://api.openfigi.com/v3/mapping", timeout=60,
                              json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in part])
            if r.status_code == 200:
                break
            time.sleep(15 * (k + 1))
        res = r.json() if r.status_code == 200 else [{}] * len(part)
        for c, x in zip(part, res):
            d = (x.get("data") or [{}])[0]
            m[c] = (d.get("ticker") or "").replace("/", "-").replace(" ", "-") if d.get("securityType2") in (
                "Common Stock", "Depositary Receipt", "Mutual Fund", None) or d.get("marketSector") == "Equity" else ""
        time.sleep(2.6)
        if i % 200 == 0:
            json.dump(m, open(f, "w"))
    json.dump(m, open(f, "w"))
    return m


def filings_13f(cik):
    os.makedirs(f"{CACHE}/13f", exist_ok=True)
    f = f"{CACHE}/13f/{cik}.json"
    if os.path.exists(f):
        return json.load(open(f))
    r = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if r is None or r.status_code != 200:
        print(f"SEC submissions {cik}: {r.status_code if r is not None else 'no answer'}")
        return None
    j = r.json()
    rec = j["filings"]["recent"]
    out = {"name": j.get("name"), "filings": []}
    for form, fd, acc, rd in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["reportDate"]):
        if form != "13F-HR" or fd < "2013-10-01":
            continue
        a = acc.replace("-", "")
        time.sleep(0.15)
        ix = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}/index.json")
        if ix is None or ix.status_code != 200:
            continue
        names = [x["name"] for x in ix.json()["directory"]["item"] if x["name"].lower().endswith(".xml")]
        names = [n for n in names if "primary_doc" not in n.lower()] or names
        hold = defaultdict(float)
        for n in names:
            time.sleep(0.15)
            x = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}/{n}")
            if x is None or x.status_code != 200 or b"infoTable" not in x.content:
                continue
            root = ET.fromstring(x.content)
            for it in root.iter():
                if not it.tag.endswith("infoTable"):
                    continue
                v = {c.tag.split("}")[1]: c for c in it.iter()}
                if "putCall" in v or (v.get("sshPrnamtType") is not None and v["sshPrnamtType"].text == "PRN"):
                    continue
                try:
                    hold[v["cusip"].text.strip().upper()] += float(v["value"].text)
                except (KeyError, ValueError, AttributeError):
                    pass
            break
        if hold:
            out["filings"].append({"filed": fd, "period": rd, "hold": hold})
    json.dump(out, open(f, "w"))
    return out


def f13(top=10):
    data = {}
    for cik, nm in FUNDS.items():
        x = filings_13f(cik)
        if x and x["filings"]:
            data[cik] = x
            print(f"{nm:28s} SEC name: {x['name']:40s} {len(x['filings'])} filings "
                  f"{min(f['filed'] for f in x['filings'])} .. {max(f['filed'] for f in x['filings'])}", flush=True)
    if not data:
        sys.exit("no 13F filings read: SEC refused (set the SEC_UA secret: 'Name email')")
    cus = [c for x in data.values() for f in x["filings"] for c, _ in
           sorted(f["hold"].items(), key=lambda kv: -kv[1])[:top + 10]]
    tick = cusip_map(cus)

    def topw(hold):
        w = defaultdict(float)
        for c, v in sorted(hold.items(), key=lambda kv: -kv[1]):
            t = tick.get(c)
            if t:
                w[t] += v
            if len(w) >= top:
                break
        s = sum(w.values())
        return {t: v / s for t, v in w.items()} if s else {}

    curves, books = {}, {}
    for cik, x in data.items():
        tg = [(f["filed"], topw(f["hold"])) for f in x["filings"]]
        eq, cov = run_book(tg)
        if eq is not None:
            curves[f"{FUNDS[cik]} (cov {cov:.0%})"] = eq
            books[cik] = tg
    # all funds together: every month, each fund's latest filed top 10, funds weighted equally
    months = pd.date_range("2014-01-01", pd.Timestamp.today(), freq="MS")
    allw, cons = [], []
    for m in months:
        ws, cnt = [], defaultdict(int)
        for cik, tg in books.items():
            last = [w for d, w in tg if pd.Timestamp(d) < m]
            if last and last[-1]:
                ws.append(last[-1])
                for t in last[-1]:
                    cnt[t] += 1
        if not ws:
            continue
        w = defaultdict(float)
        for x in ws:
            for t, v in x.items():
                w[t] += v / len(ws)
        allw.append((m - pd.Timedelta(days=1), dict(w)))
        best = sorted(cnt, key=lambda t: (-cnt[t], -w[t]))[:15]
        cons.append((m - pd.Timedelta(days=1), {t: 1 / 15 for t in best}))
    eq, cov = run_book(allw)
    curves[f"ALL funds, top 10 each (cov {cov:.0%})"] = eq
    eq, cov = run_book(cons)
    curves[f"Consensus: 15 most-held (cov {cov:.0%})"] = eq
    table(curves, f"13F copy: each fund's top {top} positions, bought the trading day after the filing is public")
    print("\nNote: funds picked in 2026 because they are famous = hindsight; GURU (etf) picks by rule, no hindsight.")


# ---------------------------------------------------------------- 3. House trade reports (PTRs)
ROW = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)\s*\[(ST)\]\s*(P|S \(partial\)|S)\s*(\d{2}/\d{2}/\d{4})\s*"
                 r"(\d{2}/\d{2}/\d{4})\s*\$([\d,]+)")


def house_trades(years):
    from pypdf import PdfReader
    os.makedirs(f"{CACHE}/house", exist_ok=True)
    rows = []
    for y in years:
        f = f"{CACHE}/house/{y}.pkl"
        if os.path.exists(f):
            rows += pd.read_pickle(f)
            continue
        r = get(f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{y}FD.zip")
        if r is None or r.status_code != 200:
            print(f"House index {y}: not reachable")
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(z.read(f"{y}FD.xml"))
        mem = [{c.tag: (c.text or "") for c in m} for m in root.findall("Member")]
        ptr = [m for m in mem if m.get("FilingType") == "P" and not m.get("DocID", "").startswith("8")]
        got, bad, yr = 0, 0, []
        for m in ptr:
            p = get(f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{y}/{m['DocID']}.pdf", tries=2)
            if p is None or p.status_code != 200:
                bad += 1
                continue
            try:
                txt = " ".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(p.content)).pages)
            except Exception:  # noqa
                bad += 1
                continue
            txt = re.sub(r"\s+", " ", txt.replace("\x00", ""))
            n = 0
            for t, _, side, td, nd, amt in ROW.findall(txt):
                yr.append(dict(member=f"{m.get('First')} {m.get('Last')}", ticker=t.replace(".", "-"),
                               side="buy" if side == "P" else "sell", traded=td, filed=m.get("FilingDate"),
                               amount=int(amt.replace(",", ""))))
                n += 1
            got += n > 0
            time.sleep(0.1)
        print(f"House {y}: {len(ptr)} electronic reports, {got} with stock rows, {bad} not read, {len(yr)} trades",
              flush=True)
        pd.to_pickle(yr, f)
        rows += yr
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df.filed, format="%m/%d/%Y", errors="coerce")
    df["traded"] = pd.to_datetime(df.traded, format="%m/%d/%Y", errors="coerce")
    return df.dropna(subset=["filed", "traded"])


def house():
    years = sorted(int(f[:4]) for f in os.listdir(f"{CACHE}/house") if f.endswith(".pkl"))
    print("House years read:", years)
    df = house_trades(years)
    df = df[(df.filed >= df.traded) & ((df.filed - df.traded).dt.days <= 400)]
    buys = df[df.side == "buy"]
    print(f"\n{len(df)} stock trades, {len(buys)} buys, {buys.ticker.nunique()} tickers, {buys.member.nunique()} members; "
          f"median delay trade -> public {int((buys.filed - buys.traded).dt.days.median())} days")
    curves = {}
    for hold in (21, 63, 126, 252):
        for big in (False, True):
            b = buys[buys.amount >= 50001] if big else buys
            curves[f"buys{' >=$50k' if big else ''}, hold {hold}d"] = house_book(b, hold)
    # the ideal: trade on the trade date itself (impossible: shows what the reporting delay costs)
    curves["(impossible) on trade date, hold 126d"] = house_book(buys.assign(filed=buys.traded), 126)
    table(curves, "House members' stock buys, bought the trading day after the report is public, equal weight")


def house_book(b, hold):
    """Each buy = one equal slice opened the day after `filed`, held `hold` trading days; the book is the average
    of the open slices (cash at the SPY return when nothing is open)."""
    idx = prices("SPY").index
    idx = idx[idx >= pd.Timestamp("2014-01-01")]
    r = pd.DataFrame(0.0, index=idx, columns=["sum", "n"])
    spy = prices("SPY").reindex(idx).pct_change().fillna(0)
    for t, g in b.groupby("ticker"):
        px = prices(t)
        if px is None:
            continue
        ret = px.reindex(idx).ffill().pct_change()
        for d in g.filed:
            i = idx.searchsorted(d + pd.Timedelta(days=1))
            if i + 1 >= len(idx):
                continue
            seg = ret.iloc[i + 1:i + 1 + hold].dropna()
            if seg.empty:
                continue
            seg = seg.copy()
            seg.iloc[0] -= COST
            seg.iloc[-1] -= COST if len(seg) == hold else 0
            r.loc[seg.index, "sum"] += seg.values
            r.loc[seg.index, "n"] += 1
    day = np.where(r.n > 0, r["sum"] / r.n.clip(lower=1), spy)
    return pd.Series(np.cumprod(1 + day), index=idx)


if __name__ == "__main__":
    if sys.argv[1] == "house-fetch":
        house_trades([int(sys.argv[2])])
    else:
        {"etf": etf, "f13": f13, "house": house}[sys.argv[1]]()
