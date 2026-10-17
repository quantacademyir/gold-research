"""Cross-check the TGJU series against two independent sources.

  * Global gold: Yahoo Finance GC=F (COMEX futures, full daily history) vs TGJU "ons".
    Futures are not spot, so levels differ by a small carry; daily returns should agree.
  * Dollar and Emami coin: bon-bast.com history page (open-market rates, Toman) vs TGJU,
    on a small random sample of dates (8 per year since mid-2021, fixed seed, 1.5 s between requests;
    the site has no free bulk download, so we do not crawl it; its history starts in
    2021-06, so 2011-2021 of the Iranian series has NO second source here).

Usage:  python3 scripts/verify_data.py
Output: data/check_gold_yahoo.csv, data/check_bonbast.csv and a printed summary.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, data=None):
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def yahoo_gold():
    url = ("https://query2.finance.yahoo.com/v8/finance/chart/GC=F"
           "?period1=967608000&period2=4102444800&interval=1d")  # range=max returns monthly bars
    res = json.loads(get(url))["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=idx).dropna()
    return s[~s.index.duplicated(keep="last")]


def bonbast(date):
    html = get("https://www.bon-bast.com/history", urllib.parse.urlencode({"date": date}).encode())
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", html, flags=re.S))
    text = re.sub(r"\s+", " ", text)
    num = r"([\d,]+)\s+([\d,]+)"
    usd = re.search(r"USD US Dollar " + num, text)
    emami = re.search(r"Emami " + num, text)
    if not (usd and emami):
        return None
    f = lambda m: tuple(10 * float(x.replace(",", "")) for x in m.groups())  # Toman -> rial
    return f(usd), f(emami)  # (sell, buy)


def main():
    load = lambda n: pd.read_csv(DATA / f"{n}.csv", parse_dates=["date"]).set_index("date")
    ons, usd, coin = load("ons_usd"), load("usd_irr"), load("emami_coin_irr")

    # 1. global gold vs Yahoo futures
    y = yahoo_gold()
    j = pd.concat({"tgju": ons.close, "yahoo": y}, axis=1).dropna()
    j = j["2000":]
    j["ratio"] = j.yahoo / j.tgju - 1
    r = np.log(j[["tgju", "yahoo"]]).diff().dropna()
    j.to_csv(DATA / "check_gold_yahoo.csv")
    print(f"GOLD  overlap {len(j)} days {j.index[0].date()}..{j.index[-1].date()}")
    print(f"  futures/TGJU - 1: mean {j.ratio.mean():+.2%}, median {j.ratio.median():+.2%}, "
          f"95% of days within {j.ratio.abs().quantile(.95):.2%}, max {j.ratio.abs().max():.2%} on {j.ratio.abs().idxmax().date()}")
    print(f"  daily log-return correlation: {r.tgju.corr(r.yahoo):.4f}")

    # 2. dollar and coin vs bon-bast (sample)
    both = pd.concat({"usd": usd.close, "coin": coin.close}, axis=1).dropna()
    both = both["2021-06":]
    rng = np.random.default_rng(7)
    dates = []
    for _, g in both.groupby(both.index.year):
        dates += list(g.index[rng.choice(len(g), size=min(8, len(g)), replace=False)])
    rows = []
    for d in sorted(dates):
        try:
            res = bonbast(d.strftime("%Y-%m-%d"))
        except Exception as e:
            print("  skip", d.date(), e)
            res = None
        time.sleep(1.5)
        if res is None:
            continue
        (us, ub), (cs, cb) = res
        rows.append(dict(date=d, tgju_usd=both.usd[d], bb_usd_sell=us, bb_usd_buy=ub,
                         tgju_coin=both.coin[d], bb_coin_sell=cs, bb_coin_buy=cb))
    c = pd.DataFrame(rows).set_index("date")
    c["usd_diff"] = c.tgju_usd / ((c.bb_usd_sell + c.bb_usd_buy) / 2) - 1
    c["coin_diff"] = c.tgju_coin / ((c.bb_coin_sell + c.bb_coin_buy) / 2) - 1
    c.to_csv(DATA / "check_bonbast.csv")
    print(f"\nIRAN  {len(c)} sampled dates with bon-bast data ({len(dates)} requested)")
    for k in ["usd_diff", "coin_diff"]:
        print(f"  {k}: median {c[k].median():+.2%}, mean abs {c[k].abs().mean():.2%}, "
              f"max abs {c[k].abs().max():.2%} on {c[k].abs().idxmax().date()}")


if __name__ == "__main__":
    main()
