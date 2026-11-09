"""Crazy strategies: invented trading rules on every market, then evolved (day desk round 33b).

No source, no theory. A generator builds strategies from ~170 building blocks: trend, momentum, RSI, breakouts, candles,
Bollinger, day levels, gaps, sweeps of yesterday's / today's high or low, a 'noise band' (price leaves its usual
distance from the day's open), IBS (where yesterday closed in its range), daily RSI2, Monday dips, the opening hour's
range, volatility regime, calendar (weekday, hour, turn of month), and other markets (Nasdaq / S&P / Bitcoin / gold /
dollar / oil / yen up or down over the last day). Every strategy gets a market, a side, 1-4 conditions, a stop (ATR,
% or none), a target, a trailing stop, an exit signal and a holding time from 1 hour to 2 weeks. All on 1-hour candles,
New York clock, 24 hours, positions held overnight and over weekends.

Then evolution: the best (on 2019-23 only) are mutated and crossed with each other for many generations. The holdout
(2024 - Sep 2026) is never used for selection; it shows whether the evolved strategies are real or curve-fitted.

Sizes: every market is scaled to Nasdaq's volatility (so a currency pair and Bitcoin count the same); costs are each
market's round trip (% of price). 'Drift' = what simply holding the market would have earned for the same time in it;
fitness = 2019-23 %/yr minus drift, so 'long Bitcoin in 2020-21' does not win by itself.

GitHub (.github/workflows/crazy.yml): `python -m research.crazy list --shard i --of n`, `build`, `run`.
"""
import argparse
import glob
import os
import random
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:          # local tests without numba
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

NY = "America/New_York"
SPLIT = pd.Timestamp("2024-01-01")
# name: (file, round-trip cost % of price) - the day desk's round-9 market table
M = {"US Tech 100": ("NSXUSD", .008), "US 500": ("SPXUSD", .012), "Germany 40": ("GRXEUR", .015),
     "UK 100": ("UKXGBP", .02), "France 40": ("FRXEUR", .025), "Europe 50": ("ETXEUR", .03),
     "Japan 225": ("JPXJPY", .03), "Australia 200": ("AUXAUD", .025), "Hong Kong 50": ("HKXHKD", .04),
     "US Dollar Index": ("UDXUSD", .02), "Gold": ("XAUUSD", .02), "Silver": ("XAGUSD", .08),
     "Brent oil": ("BCOUSD", .05), "WTI oil": ("WTIUSD", .05), "Bitcoin": ("BTCUSDT", .15), "Ethereum": ("ETHUSDT", .2),
     "Solana": ("SOLUSDT", .4), "XRP": ("XRPUSDT", .4), "Litecoin": ("LTCUSDT", .4), "Cardano": ("ADAUSDT", .4),
     "Dogecoin": ("DOGEUSDT", .4), "BNB": ("BNBUSDT", .4)}
for _p in "EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD".split():
    M[_p] = (_p, .01)
for _p in ("EURGBP EURJPY GBPJPY EURCHF AUDJPY EURAUD GBPCHF CADJPY CHFJPY AUDNZD EURCAD GBPAUD AUDCAD NZDJPY GBPCAD "
           "GBPNZD EURNZD AUDCHF CADCHF NZDCAD NZDCHF").split():
    M[_p] = (_p, .02)
for _p in "USDMXN USDZAR USDNOK USDSEK EURNOK EURSEK USDPLN EURPLN USDTRY USDHUF USDCZK USDSGD".split():
    M[_p] = (_p, .06)
REFS = {"Nasdaq": "NSXUSD", "S&P": "SPXUSD", "Bitcoin": "BTCUSDT", "gold": "XAUUSD", "dollar": "UDXUSD",
        "oil": "WTIUSD", "USDJPY": "USDJPY"}
ADJ = ("Amber Brass Cobalt Copper Crimson Dusk Ember Frost Granite Hollow Indigo Jade Lunar Marble Neon Onyx Polar "
       "Quiet Rusty Silent Solar Tidal Velvet Wild Zinc Iron Golden Hazel Scarlet Storm Feral Cosmic Toxic Turbo "
       "Haunted Electric Drunken Rabid Atomic Chaos Mad Laser Plasma Vortex Nuclear Savage Psycho Hyper Rogue").split()
NOUN = ("Falcon Otter Viper Heron Badger Lynx Raven Marlin Bison Gecko Jackal Kestrel Mantis Narwhal Ocelot Panther "
        "Quail Stoat Tapir Walrus Yak Cobra Moth Shark Wolf Hare Crane Beetle Puma Orca Goblin Kraken Hydra Yeti "
        "Wombat Platypus Llama Honeybadger Piranha Komodo Axolotl Scorpion Mongoose Wolverine Pelican Hyena").split()
HOLDS = (1, 2, 4, 8, 16, 24, 48, 72, 120, 240, 336)       # max holding time, hours (1-hour candles)


# ----------------------------------------------------------------------------------------------------------- data
def markets(shard, of):
    return [M[m][0] for k, m in enumerate(sorted(M)) if k % of == shard]


def build(raw, sym, out):
    """1-minute zips (HistData: New York clock; Binance: UTC ms) -> 1-hour candles on the New York clock."""
    frames = []
    for z in sorted(glob.glob(f"{raw}/*.zip")):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        for n in zf.namelist():
            if not n.lower().endswith(".csv"):
                continue
            with zf.open(n) as f:
                if sym.endswith("USDT"):
                    df = pd.read_csv(f, header=None, usecols=range(5), names=["t", "open", "high", "low", "close"])
                    t = df.t.astype("int64")
                    t = np.where(t > 1e14, t // 1000, t)
                    df.index = pd.to_datetime(t, unit="ms", utc=True).tz_convert(NY)
                else:
                    df = pd.read_csv(f, sep=";", header=None, names=["t", "open", "high", "low", "close", "v"])
                    df.index = pd.to_datetime(df.t, format="%Y%m%d %H%M%S").dt.tz_localize(
                        NY, ambiguous="NaT", nonexistent="NaT")
                    df = df[df.index.notna()]
                frames.append(df[["open", "high", "low", "close"]])
    if not frames:
        print("no data", sym)
        return
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated()]
    b = m1.resample("60min", label="left", closed="left", offset="30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b = b[b.index >= "2019-01-01"]
    b.index = b.index.tz_localize(None)
    print(sym, b.index[0], b.index[-1], len(b), flush=True)
    b.to_pickle(out)


# ------------------------------------------------------------------------------------------------- building blocks
def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def rsi(x, n):
    d = x.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def blocks(b, refs):
    """Every building block as name -> (long version, short version), boolean arrays on the candle close."""
    c, o, h, l = b.close, b.open, b.high, b.low
    hr = pd.Series(b.index.hour, b.index)
    tdate = pd.Series((b.index + pd.Timedelta(hours=6)).normalize(), b.index)    # day starts 18:00 NY
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(24, min_periods=5).mean()
    g = b.groupby(tdate.values)
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    pd_ = d.shift(1)
    m = lambda s: pd.Series(tdate.map(s).values, b.index)
    dopen, pdh, pdl, pdc = m(d.open), m(pd_.high), m(pd_.low), m(pd_.close)
    out = {}

    def add(name, lo, sh):
        out[name] = (np.asarray(pd.Series(lo).fillna(False), bool), np.asarray(pd.Series(sh).fillna(False), bool))

    for n in (8, 20, 50, 100, 200):
        e = ema(c, n)
        add(f"close above EMA{n}", c > e, c < e)
        add(f"close below EMA{n} (fade)", c < e, c > e)
    for a, z in ((8, 21), (20, 50), (50, 200)):
        ea, ez = ema(c, a), ema(c, z)
        add(f"EMA{a} above EMA{z}", ea > ez, ea < ez)
        add(f"EMA{a} crosses EMA{z}", (ea > ez) & (ea.shift() <= ez.shift()), (ea < ez) & (ea.shift() >= ez.shift()))
    for n in (20, 50, 200):
        s = m(d.close.rolling(n).mean().shift(1))
        add(f"above daily SMA{n}", c > s, c < s)
    for n in (2, 14):
        r = rsi(c, n)
        for lvl in ((5, 10, 25) if n == 2 else (25, 30, 40, 50)):
            add(f"RSI{n} below {lvl}", r < lvl, r > 100 - lvl)
        add(f"RSI{n} above {80 if n == 2 else 60} (momentum)", r > (80 if n == 2 else 60), r < (20 if n == 2 else 40))
    dr2 = m(rsi(d.close, 2).shift(1))
    add("daily RSI2 below 10 yesterday", dr2 < 10, dr2 > 90)
    ibs = m(((d.close - d.low) / (d.high - d.low)).shift(1))
    add("IBS yesterday < 0.2", ibs < 0.2, ibs > 0.8)
    add("IBS yesterday > 0.8 (momentum)", ibs > 0.8, ibs < 0.2)
    cibs = (c - l) / (h - l)
    add("candle closes in its bottom 20%", cibs < 0.2, cibs > 0.8)
    for k in (1, 4, 24, 120):
        z = (c / c.shift(k) - 1) / (atr / c * np.sqrt(k / 1.0))
        add(f"up over last {k}h", z > 0.5, z < -0.5)
        add(f"down hard over last {k}h (fade)", z < -1.5, z > 1.5)
    for n in (6, 24, 120, 480):
        hh, ll = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
        add(f"breaks {n}h high", c > hh, c < ll)
        add(f"at {n}h low (fade)", c < ll, c > hh)
    d20h, d20l = m(d.high.rolling(20).max().shift(1)), m(d.low.rolling(20).min().shift(1))
    add("breaks 20-day high", c > d20h, c < d20l)
    for k in (2, 3, 4):
        up, dn = (c > o).rolling(k).sum() == k, (c < o).rolling(k).sum() == k
        add(f"{k} up candles in a row", up, dn)
        add(f"{k} down candles in a row (fade)", dn, up)
    big = (h - l) > 2 * atr
    add("big candle up", big & (c > o), big & (c < o))
    add("big candle down (fade)", big & (c < o), big & (c > o))
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    add("above upper Bollinger", c > m20 + 2 * s20, c < m20 - 2 * s20)
    add("below lower Bollinger (fade)", c < m20 - 2 * s20, c > m20 + 2 * s20)
    add("above yesterday's high", c > pdh, c < pdl)
    add("below yesterday's low (fade)", c < pdl, c > pdh)
    add("above yesterday's close", c > pdc, c < pdc)
    add("above today's open", c > dopen, c < dopen)
    add("below today's open (fade)", c < dopen, c > dopen)
    add("sweep of yesterday's low, closes back above", (l < pdl) & (c > pdl), (h > pdh) & (c < pdh))
    run_lo = g.low.cummin().groupby(tdate.values).shift(1)
    run_hi = g.high.cummax().groupby(tdate.values).shift(1)
    add("sweep of today's low, closes back above", (l < run_lo) & (c > run_lo), (h > run_hi) & (c < run_hi))
    gap = (dopen / pdc - 1) * 100
    add("day opened up", gap > 0.1, gap < -0.1)
    add("day opened down (fade)", gap < -0.1, gap > 0.1)
    # noise band: usual distance from the day's open at this hour over the last 14 days
    mv = (c / dopen - 1).abs()
    piv = pd.DataFrame({"d": tdate.values, "h": hr.values, "mv": mv.values}).pivot_table(index="d", columns="h", values="mv")
    sig = piv.rolling(14, min_periods=7).mean().shift(1).stack()
    sg = pd.Series(sig.reindex(list(zip(tdate.values, hr.values))).values, b.index)
    up_b, lo_b = np.maximum(dopen, pdc) * (1 + sg), np.minimum(dopen, pdc) * (1 - sg)
    add("noise band break", c > up_b, c < lo_b)
    add("noise band break (fade)", c < lo_b, c > up_b)
    first = g.head(1).index
    orh = m(b.loc[first].set_axis(tdate[first].values).high)
    orl = m(b.loc[first].set_axis(tdate[first].values).low)
    add("breaks the day's first hour", c > orh, c < orl)
    ny_or = b[b.index.hour == 9]
    nyh, nyl = m(ny_or.set_axis(tdate[ny_or.index].values).high), m(ny_or.set_axis(tdate[ny_or.index].values).low)
    late = hr >= 10
    add("breaks the 9:30 NY hour's range", late & (c > nyh), late & (c < nyl))
    wd = pd.Series(tdate.dt.dayofweek.values, b.index)
    for k, nm in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "the weekend"]):
        x = (wd == k) if k < 5 else (wd >= 5)
        add(f"on {nm}", x, x)
    r14 = rsi(c, 14)
    add("Monday dip (RSI14 < 50)", (wd == 0) & (r14 < 50), (wd == 0) & (r14 > 50))
    for nm, (a, z) in {"Asia hours 20-03": (20, 3), "London 03-09": (3, 9), "NY morning 09-12": (9, 12),
                       "NY afternoon 12-16": (12, 16), "evening 16-20": (16, 20)}.items():
        x = (hr >= a) | (hr < z) if a > z else (hr >= a) & (hr < z)
        add(f"in {nm}", x, x)
    for hh_ in (3, 9, 10, 15):
        add(f"at {hh_}:30 NY candle close", hr == hh_, hr == hh_)
    dom = pd.Series(tdate.dt.day.values, b.index)
    add("turn of the month", (dom >= 27) | (dom <= 3), (dom >= 27) | (dom <= 3))
    vr = tr.rolling(24).mean() / tr.rolling(240).mean()
    add("volatile (24h range 1.3x usual)", vr > 1.3, vr > 1.3)
    add("calm (24h range 0.8x usual)", vr < 0.8, vr < 0.8)
    rng_d = d.high - d.low
    add("yesterday narrowest of 7 days", m((rng_d == rng_d.rolling(7).min()).shift(1)), m((rng_d == rng_d.rolling(7).min()).shift(1)))
    for nm, s in refs.items():
        r = s.reindex(b.index, method="ffill")
        add(f"{nm} up over last 24h", r > 0, r < 0)
        add(f"{nm} down over last 24h (fade)", r < 0, r > 0)
    return out, np.asarray(atr.bfill(), float)


# --------------------------------------------------------------------------------------------------------- engine
@njit(cache=True)
def sim(o, h, l, c, le, se, xl, xs, sd, tgt, trail, maxh, cost, scale, split):
    """One position at a time, entry on the candle close, stop first if a candle hits stop and target.
    Returns per period (0 = 2019-23, 1 = 2024+): sum %, trades, wins %, losses %, net long candles."""
    r = np.zeros((2, 5))
    pos, ent, sl, tp, ei = 0, 0.0, np.nan, np.nan, 0
    for i in range(len(c)):
        p = 0 if i < split else 1
        if pos != 0:
            x = np.nan
            if pos == 1:
                if l[i] <= sl:
                    x = min(sl, o[i])
                elif h[i] >= tp:
                    x = max(tp, o[i])
            else:
                if h[i] >= sl:
                    x = max(sl, o[i])
                elif l[i] <= tp:
                    x = min(tp, o[i])
            if np.isnan(x) and ((pos == 1 and xl[i]) or (pos == -1 and xs[i]) or i - ei >= maxh):
                x = c[i]
            if np.isnan(x) and trail > 0:
                if pos == 1:
                    sl = max(sl, c[i] - trail * sd[i]) if not np.isnan(sl) else c[i] - trail * sd[i]
                else:
                    sl = min(sl, c[i] + trail * sd[i]) if not np.isnan(sl) else c[i] + trail * sd[i]
            r[p, 4] += pos
            if not np.isnan(x):
                q = 0 if ei < split else 1
                pct = (pos * (x / ent - 1) * 100 - cost) * scale
                r[q, 0] += pct
                r[q, 1] += 1
                if pct > 0:
                    r[q, 2] += pct
                else:
                    r[q, 3] -= pct
                pos = 0
                continue
        if pos == 0 and (le[i] or se[i]) and i < len(c) - 1:
            pos = 1 if le[i] else -1
            ent, ei = c[i], i
            sl, tp = np.nan, np.nan
            if sd[i] > 0:
                sl = ent - pos * sd[i]
                if tgt > 0:
                    tp = ent + pos * tgt * sd[i]
    return r


# ------------------------------------------------------------------------------------------------------ strategies
_D = {}    # market -> dict(o, h, l, c, bl, atr, cost, scale, split, yrs, drift)
_NAMES = []


def setup(data_dir):
    if _D:
        return
    raw = {}
    for f in sorted(glob.glob(f"{data_dir}/**/*.pkl", recursive=True)):
        raw[os.path.basename(f)[:-4]] = pd.read_pickle(f)
    refs = {}
    for nm, sym in REFS.items():
        if sym in raw:
            x = raw[sym].close
            refs[nm] = x / x.shift(24) - 1
    vol = lambda b: (b.close[b.index < SPLIT].resample("D").last().dropna().pct_change().std()
                     * np.sqrt(len(b[b.index < SPLIT].resample("D").last().dropna()) / 5.0) * 100)
    target = vol(raw["NSXUSD"]) if "NSXUSD" in raw else 22.0
    for mk, (sym, cost) in M.items():
        if sym not in raw or len(raw[sym]) < 5000:
            continue
        b = raw[sym]
        bl, atr = blocks(b, refs)
        split = int(np.searchsorted(b.index.values, np.datetime64(SPLIT)))
        t = b.index
        yrs = ((t[split - 1] - t[0]).days / 365.25, (t[-1] - t[split]).days / 365.25)
        lr = np.log(b.close).diff().fillna(0).values
        drift = (lr[:split].mean() * 100, lr[split:].mean() * 100)      # % per candle held, per period
        _D[mk] = dict(o=b.open.values, h=b.high.values, l=b.low.values, c=b.close.values, bl=bl, atr=atr,
                      cost=cost, scale=float(target / vol(b)), split=split, yrs=yrs, drift=drift,
                      bh=((b.close.iat[split - 1] / b.close.iat[0] - 1) * 100, (b.close.iat[-1] / b.close.iat[split] - 1) * 100))
    _NAMES[:] = list(next(iter(_D.values()))["bl"])


def name(rng, k):
    return f"{rng.choice(ADJ)} {rng.choice(NOUN)} #{k}"


def draw(rng, k):
    return dict(id=k, name=name(rng, k), gen=0, market=rng.choice(sorted(_D)),
                conds=rng.sample(_NAMES, rng.choice((1, 2, 2, 3, 3, 4))),
                side=rng.choice(("long", "long", "short", "both")),
                stop=rng.choice((("atr", 1.0), ("atr", 2.0), ("atr", 4.0), ("atr", 8.0), ("pct", 0.5), ("pct", 1.0),
                                 ("pct", 2.0), ("pct", 5.0), (None, 0))),
                tgt=rng.choice((0, 0, 1.0, 1.5, 2.0, 3.0, 5.0)), trail=rng.choice((0, 0, 0, 1.0, 2.0, 4.0)),
                hold=rng.choice(HOLDS), exit=rng.choice([None, None] + _NAMES))


def mutate(rng, s, k, gen):
    s = dict(s, conds=list(s["conds"]), id=k, gen=gen, name=name(rng, k), parent=s["name"])
    for _ in range(rng.choice((1, 1, 2, 3))):
        what = rng.choice(("cond", "cond", "add", "drop", "market", "side", "stop", "tgt", "trail", "hold", "exit"))
        if what == "cond":
            s["conds"][rng.randrange(len(s["conds"]))] = rng.choice(_NAMES)
        elif what == "add" and len(s["conds"]) < 5:
            s["conds"].append(rng.choice(_NAMES))
        elif what == "drop" and len(s["conds"]) > 1:
            s["conds"].pop(rng.randrange(len(s["conds"])))
        elif what == "market":
            s["market"] = rng.choice(sorted(_D))
        else:
            t = draw(rng, k)
            key = {"side": "side", "stop": "stop", "tgt": "tgt", "trail": "trail", "hold": "hold", "exit": "exit"}.get(what)
            if key:
                s[key] = t[key]
    s["conds"] = list(dict.fromkeys(s["conds"]))
    return s


def cross(rng, a, b, k, gen):
    pool = list(dict.fromkeys(a["conds"] + b["conds"]))
    s = dict(a, id=k, gen=gen, name=name(rng, k), parent=f"{a['name']} x {b['name']}",
             conds=rng.sample(pool, min(len(pool), rng.choice((1, 2, 3, 4)))))
    for key in ("market", "side", "stop", "tgt", "trail", "hold", "exit"):
        if rng.random() < 0.5:
            s[key] = b[key]
    return s


def key(s):
    return (s["market"], tuple(sorted(s["conds"])), s["side"], tuple(s["stop"]), s["tgt"], s["trail"], s["hold"], s["exit"])


def rules(s):
    st = {None: "no stop", "atr": f"stop {s['stop'][1]}x ATR(24h)", "pct": f"stop {s['stop'][1]}%"}[s["stop"][0]]
    return (f"{s['market']} {s['side']}: " + " AND ".join(s["conds"]) + f"; {st}"
            + (f"; target {s['tgt']}R" if s["tgt"] and s["stop"][0] else "")
            + (f"; trailing stop {s['trail']}x ATR" if s["trail"] else "")
            + (f"; exit on '{s['exit']}' (other side)" if s["exit"] else "") + f"; max {s['hold']}h")


def run(s):
    d = _D[s["market"]]
    lo = np.logical_and.reduce([d["bl"][n][0] for n in s["conds"]])
    sh = np.logical_and.reduce([d["bl"][n][1] for n in s["conds"]])
    z = np.zeros(len(lo), bool)
    le = lo if s["side"] in ("long", "both") else z
    se = sh & ~le if s["side"] in ("short", "both") else z
    if s["exit"]:
        xl, xs = d["bl"][s["exit"]][1], d["bl"][s["exit"]][0]
    else:
        xl = xs = z
    kind, x = s["stop"]
    sd = d["atr"] * x if kind == "atr" else d["c"] * x / 100 if kind == "pct" else np.zeros(len(lo))
    trail = float(s["trail"]) if kind or s["trail"] else 0.0
    if not kind and s["trail"]:
        sd = d["atr"] * 1.0
        trail = float(s["trail"])
    r = sim(d["o"], d["h"], d["l"], d["c"], le, se, xl, xs, sd, float(s["tgt"] if kind else 0), trail,
            int(s["hold"]), d["cost"], d["scale"], d["split"])
    out = dict(id=s["id"], gen=s["gen"], name=s["name"], parent=s.get("parent", ""), market=s["market"], rules=rules(s))
    for p, per in enumerate(("train", "hold")):
        y = d["yrs"][p]
        out[f"{per}_pct_yr"] = round(r[p, 0] / y, 2)
        out[f"{per}_n_yr"] = round(r[p, 1] / y, 1)
        out[f"{per}_pf"] = round(r[p, 2] / r[p, 3], 2) if r[p, 3] > 0 else np.nan
        out[f"{per}_drift_yr"] = round(r[p, 4] * d["drift"][p] * d["scale"] / y, 2)
        out[f"{per}_excess_yr"] = round(out[f"{per}_pct_yr"] - out[f"{per}_drift_yr"], 2)
        out[f"{per}_gross_excess_yr"] = round(out[f"{per}_excess_yr"] + r[p, 1] * d["cost"] * d["scale"] / y, 2)
    return out


def fitness(r):
    """2019-23 only: %/yr above drift, needs at least 10 trades a year."""
    return r.train_excess_yr.where(r.train_n_yr >= 10, -1e9)


def evolve(data, out, n0, pop, gens, seed, workers):
    os.makedirs(out, exist_ok=True)
    setup(data)
    print(f"{len(_D)} markets, {len(_NAMES)} building blocks", flush=True)
    rng = random.Random(seed)
    seen, rows, k = {}, [], 0

    def evaluate(p, strats):
        new = []
        for s in strats:
            kk = key(s)
            if kk not in seen and s["conds"]:
                seen[kk] = s
                new.append(s)
        res = p.map(run, new, chunksize=64)
        rows.extend(res)
        return pd.DataFrame(res).assign(_s=new) if res else pd.DataFrame()

    log = []
    with Pool(workers, initializer=setup, initargs=(data,)) as p:
        strats = []
        for _ in range(n0):
            k += 1
            strats.append(draw(rng, k))
        g0 = evaluate(p, strats)
        g0["fit"] = fitness(g0)
        popu = g0.sort_values("fit", ascending=False).head(pop)
        for gen in range(1, gens + 1):
            elite = popu.head(pop // 5)
            kids = []
            par = list(popu._s)
            w = np.linspace(2, 0.2, len(par))
            for _ in range(pop):
                k += 1
                a = rng.choices(par, weights=w)[0]
                if rng.random() < 0.35:
                    kids.append(cross(rng, a, rng.choices(par, weights=w)[0], k, gen))
                else:
                    kids.append(mutate(rng, a, k, gen))
            ch = evaluate(p, kids)
            if len(ch):
                ch["fit"] = fitness(ch)
            popu = pd.concat([elite, ch]).drop_duplicates("id").sort_values("fit", ascending=False).head(pop)
            top = popu.head(50)
            log.append(dict(gen=gen, evaluated=len(rows), best_train=popu.train_excess_yr.iat[0],
                            top50_train_pct=top.train_pct_yr.mean(), top50_hold_pct=top.hold_pct_yr.mean(),
                            top50_train_excess=top.train_excess_yr.mean(), top50_hold_excess=top.hold_excess_yr.mean(),
                            top50_hold_positive=(top.hold_pct_yr > 0).mean()))
            print(pd.DataFrame(log).tail(1).round(2).to_string(index=False, header=gen == 1), flush=True)
    allr = pd.DataFrame(rows)
    allr.to_csv(f"{out}/crazy_all.csv.gz", index=False)
    report(allr, popu.drop(columns="_s"), pd.DataFrame(log), out)


def report(a, popu, log, out):
    L = []
    p = lambda *x: L.append(" ".join(str(y) for y in x))
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 200)
    cols = ["name", "train_pct_yr", "train_excess_yr", "train_pf", "train_n_yr", "hold_pct_yr", "hold_excess_yr",
            "hold_pf", "hold_n_yr", "rules"]
    g0 = a[(a.gen == 0) & (a.train_n_yr >= 10) & (a.hold_n_yr >= 10)]
    tp, hp = (g0.train_pct_yr > 0).mean(), (g0.hold_pct_yr > 0).mean()
    p(f"Crazy strategies (round 33b): {len(a)} strategies tested on {a.market.nunique()} markets, 1h candles, "
      f"held 1h-2 weeks; sizes scaled to Nasdaq's volatility; 2019-23 = selection, 2024 - Sep 2026 = holdout")
    p(f"\n1) {len(g0)} purely random ones (>= 10 trades a year): positive 2019-23 {tp:.1%}, 2024+ {hp:.1%}, both "
      f"{((g0.train_pct_yr > 0) & (g0.hold_pct_yr > 0)).mean():.1%} (chance if unrelated {tp * hp:.1%}); "
      f"rank correlation 2019-23 vs 2024+ {g0.train_pct_yr.rank().corr(g0.hold_pct_yr.rank()):+.3f}, "
      f"above drift {g0.train_excess_yr.rank().corr(g0.hold_excess_yr.rank()):+.3f}, above drift before costs "
      f"{g0.train_gross_excess_yr.rank().corr(g0.hold_gross_excess_yr.rank()):+.3f} (costs alone make busy strategies "
      f"lose in both periods; the last number is the one that says whether the past predicts the future)")
    g0s = g0.sort_values("train_excess_yr", ascending=False)
    for k in (50, len(g0s) // 100, len(g0s) // 10):
        t = g0s.head(k)
        p(f"   best {k} random on 2019-23: {t.train_pct_yr.mean():+.1f} %/yr ({t.train_excess_yr.mean():+.1f} above drift) "
          f"-> 2024+ {t.hold_pct_yr.mean():+.1f} %/yr ({t.hold_excess_yr.mean():+.1f}), {(t.hold_pct_yr > 0).mean():.0%} positive")
    p(f"   all random: 2024+ {g0.hold_pct_yr.mean():+.1f} %/yr ({g0.hold_excess_yr.mean():+.1f} above drift)")
    p("\n2) Evolution (selection on 2019-23 only; watch the 2024+ columns):")
    p(log.round(2).to_string(index=False))
    fin = popu[(popu.fit > -1e8) & (popu.hold_n_yr >= 5)]
    p(f"\n3) Final population ({len(fin)}): 2019-23 {fin.train_pct_yr.mean():+.1f} %/yr ({fin.train_excess_yr.mean():+.1f} above "
      f"drift) -> 2024+ {fin.hold_pct_yr.mean():+.1f} %/yr ({fin.hold_excess_yr.mean():+.1f}); "
      f"positive on 2024+: {(fin.hold_pct_yr > 0).mean():.0%}, above drift: {(fin.hold_excess_yr > 0).mean():.0%}")
    p("   markets in the final population: " + ", ".join(f"{m} {n}" for m, n in fin.market.value_counts().head(12).items()))
    p("\nTop 30 evolved (by 2019-23) and their 2024+:")
    p(fin.drop_duplicates(["market", "train_pct_yr", "hold_pct_yr"]).head(30)[cols].to_string(index=False))
    ok = a[(a.train_n_yr >= 10) & (a.hold_n_yr >= 10)]
    s = ok.assign(w=ok[["train_excess_yr", "hold_excess_yr"]].min(axis=1)).sort_values("w", ascending=False)
    p(f"\nBest on BOTH periods above drift, out of all {len(ok)} (picked by looking at the holdout = luck included):")
    p(s.drop_duplicates(["market", "train_pct_yr", "hold_pct_yr"]).head(30)[cols].to_string(index=False))
    txt = "\n".join(L)
    print(txt)
    open(f"{out}/crazy_report.txt", "w").write(txt + "\n")
    popu.to_csv(f"{out}/crazy_final_population.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "build", "run"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--raw")
    ap.add_argument("--sym")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n0", type=int, default=30000)
    ap.add_argument("--pop", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.cmd == "list":
        for sym in markets(a.shard, a.of):
            print(("binance " if sym.endswith("USDT") else "histdata ") + sym)
    elif a.cmd == "build":
        build(a.raw, a.sym, a.out)
    else:
        evolve(a.data, a.out, a.n0, a.pop, a.gens, a.seed, a.workers)
