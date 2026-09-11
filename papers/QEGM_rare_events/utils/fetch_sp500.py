"""Fetch S&P 500 daily closes and derive the packaged log-return CSV.

Downloads the ^GSPC daily history (1990-01-01 .. 2022-12-31, the range
named in the paper's Sec. VI.D finance experiment) from the public Yahoo
Finance chart API, computes daily log-returns from closing prices, and
writes ``data/QEGM_rare_events/sp500_daily_logreturns_1990_2022.csv`` at
the repository root. The derived CSV is committed so the experiment runs
from a fresh clone without network access; run this script only to
regenerate it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import urllib.request
from pathlib import Path

PERIOD1 = 631152000  # 1990-01-01 UTC
PERIOD2 = 1672531200  # 2023-01-01 UTC
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
    f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "QEGM_rare_events"
    / "sp500_daily_logreturns_1990_2022.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        payload = json.load(resp)

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    series = [
        (dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat(), c)
        for ts, c in zip(timestamps, closes)
        if c is not None
    ]
    if len(series) < 1000:
        raise RuntimeError(f"Suspiciously few rows from Yahoo: {len(series)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "log_return"])
        for (_, prev), (date, close) in zip(series, series[1:]):
            writer.writerow([date, f"{math.log(close / prev):.10f}"])
    print(
        f"wrote {args.out} ({len(series) - 1} log-returns, "
        f"{series[0][0]} .. {series[-1][0]})"
    )


if __name__ == "__main__":
    main()
