"""Second source for the Iranian dollar and Emami coin BEFORE 2021-06, from the Internet
Archive's snapshots of bonbast.com (the live site's rates at the moment of each snapshot).

Each snapshot is a point-in-time quote taken at an unknown hour, so we compare it with the
TGJU close of the snapshot day AND of the previous trading day, and call it "consistent"
if it is within 3% of either. The rule was fixed before looking at the results.

Usage:  python3 scripts/verify_wayback.py        (about 150 requests, 1 s apart)
Output: data/check_wayback.csv and a printed summary.
"""
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}
CDX = ("https://web.archive.org/cdx/search/cdx?url=bonbast.com&from=2012&to=2021"
       "&output=json&filter=statuscode:200&collapse=timestamp:8")


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")


def parse(html):
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", html, flags=re.S)))
    usd = re.search(r"USD US Dollar ([\d,]+) ([\d,]+)", text)
    emami = re.search(r"\bEmami ([\d,]+) ([\d,]+)", text)
    if not (usd and emami):
        return None
    mid = lambda m: 10 * (float(m.group(1).replace(",", "")) + float(m.group(2).replace(",", ""))) / 2  # Toman -> rial
    return mid(usd), mid(emami)


def main():
    snaps = json.loads(get(CDX))[1:]
    load = lambda n: pd.read_csv(DATA / f"{n}.csv", index_col=0, parse_dates=True).close
    tg = pd.concat({"usd": load("usd_irr"), "coin": load("emami_coin_irr")}, axis=1).dropna()
    rows, failed = [], 0
    for _, ts, orig, *_ in snaps:
        day = pd.Timestamp(ts[:8])
        if day >= pd.Timestamp("2021-06-01"):
            continue
        try:
            res = parse(get(f"https://web.archive.org/web/{ts}id_/{orig}"))
        except Exception:
            res = None
        time.sleep(1)
        if res is None:
            failed += 1
            continue
        same = tg[tg.index <= day]
        if len(same) < 2:
            continue
        for k, v in zip(("usd", "coin"), res):
            a, b = same[k].iloc[-1], same[k].iloc[-2]  # close on/before the day, and the one before
            # Early snapshots (2013) show rial, later ones Toman, and we multiply by 10 for Toman.
            # A factor-of-10 gap cannot be a price move, so rescale those and say so.
            rescaled = v / a > 5
            if rescaled:
                v /= 10
            rows.append(dict(snapshot=ts, series=k, bonbast=v, tgju_day=a, tgju_prev=b, rescaled=rescaled,
                             diff=min((v / a - 1, v / b - 1), key=abs)))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "check_wayback.csv", index=False)
    print(f"{len(snaps)} snapshots listed, {out.snapshot.nunique()} parsed, {failed} could not be parsed")
    print(f"  rescaled by 10 (unit shown in rial): {int(out.rescaled.sum() / 2)} snapshots")
    for k, g in out.groupby("series"):
        print(f"  {k}: n={len(g)}, median diff {g['diff'].median():+.2%}, within 3%: {(g['diff'].abs() < .03).mean():.0%}, "
              f"worst {g['diff'].abs().max():.1%} at {g.loc[g['diff'].abs().idxmax(), 'snapshot'][:8]}")


if __name__ == "__main__":
    main()
