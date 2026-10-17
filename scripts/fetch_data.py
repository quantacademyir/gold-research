"""Download the daily series for the coin-bubble study and save them as CSV.

Source: TGJU (tgju.org), an Iranian price-information site. It is NOT an official
source: values are as published by the site, in rial for Iranian series and in USD
for the global ounce. Check the data before relying on it (see notebooks/01).

Usage:  python3 scripts/fetch_data.py
Output: data/ons_usd.csv, data/usd_irr.csv, data/emami_coin_irr.csv
Columns: date, open, low, high, close
"""
import csv
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/{}"
SERIES = {  # tgju name -> output file
    "ons": "ons_usd.csv",  # global gold, USD per troy ounce
    "price_dollar_rl": "usd_irr.csv",  # open-market dollar, rial per USD
    "sekee": "emami_coin_irr.csv",  # Emami coin, rial per coin
}
DATA = Path(__file__).resolve().parent.parent / "data"


def num(text):
    return float(text.replace(",", ""))


def fetch(name):
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


def main():
    DATA.mkdir(exist_ok=True)
    for name, filename in SERIES.items():
        rows = []
        for r in fetch(name):
            # r = [open, low, high, close, change, change%, gregorian date, jalali date]
            date = r[6].replace("/", "-")
            rows.append((date, num(r[0]), num(r[1]), num(r[2]), num(r[3])))
        rows.sort()
        with open(DATA / filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "low", "high", "close"])
            w.writerows(rows)
        print(f"{filename}: {len(rows)} rows, {rows[0][0]} to {rows[-1][0]}")


if __name__ == "__main__":
    sys.exit(main())
