"""Download the daily Yahoo Finance price histories used by arXiv:2601.03802.

The paper retrieves every series through the Yahoo Finance API via
``yfinance`` (footnote 1, https://github.com/ranaroussi/yfinance). This script
calls the same public Yahoo chart endpoint that ``yfinance`` wraps and stores,
for each symbol, the unmodified JSON response and a CSV with
``date, open, high, low, close, adj_close, volume``. ``adj_close`` is Yahoo's
split- and dividend-adjusted close (the paper's input for single stocks);
``close`` is the raw close (the paper's input for the S&P 500).

Yahoo revises adjusted prices over time, so a fresh download is a new
snapshot, not the paper's exact data. ``manifest.json`` records the fetch
time, URL, row count, date range and SHA-256 of every file.

Usage (from the repository root)::

    python papers/QML_financial_benchmark/utils/download_yahoo_data.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

# Table III (U.S. universe), Table IV (Turkish universe) and Table V (7-D
# cross-asset indices). The 64-D set uses "eight major indices" that the paper
# does not list; Table V's seven plus ^GSPC cover the documented assumption.
SYMBOLS = {
    "us_equities": ["^GSPC", "AAPL", "BA", "GILD", "DVN", "LNC"],
    "turkish_equities": ["KCHOL.IS", "GARAN.IS", "TUPRS.IS", "ULKER.IS", "TCELL.IS"],
    "global_indices": ["^N225", "^HSI", "^AORD", "^GDAXI", "^FTSE", "^DJI", "^NYA"],
}
# Covers Fig. 6's expanding windows (training from about 2003), Task 1
# (2009-2021), Task 2 regimes F1-F4 (2008-01-02 to 2024-12-31) and Task 3.
DEFAULT_START = "2000-01-01"
DEFAULT_END_EXCLUSIVE = "2025-01-01"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "QML_financial_benchmark"
    / "yahoo_daily_prices"
)


def _utc_timestamp(day: str) -> int:
    return int(
        dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp()
    )


def fetch_chart(symbol: str, start: str, end_exclusive: str) -> tuple[str, bytes]:
    """Fetch one symbol's daily chart JSON from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Yahoo Finance symbol, for example ``"KCHOL.IS"`` or ``"^GSPC"``.
    start : str
        First calendar day to request, ISO format.
    end_exclusive : str
        Day after the last calendar day to request, ISO format.

    Returns
    -------
    tuple[str, bytes]
        The request URL and the raw response body.
    """
    query = urllib.parse.urlencode(
        {
            "period1": _utc_timestamp(start),
            "period2": _utc_timestamp(end_exclusive),
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
    )
    url = CHART_URL.format(symbol=urllib.parse.quote(symbol), query=query)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return url, response.read()


def chart_to_rows(payload: dict) -> list[dict]:
    """Convert a Yahoo chart payload to daily rows keyed by exchange-local date.

    Parameters
    ----------
    payload : dict
        Parsed chart JSON.

    Returns
    -------
    list[dict]
        One row per timestamp. Missing Yahoo values stay ``None``.

    Raises
    ------
    ValueError
        If Yahoo returned an error or no result.
    """
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    result = chart["result"][0]
    exchange_zone = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    quote = result["indicators"]["quote"][0]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    rows = []
    for index, timestamp in enumerate(result["timestamp"]):
        rows.append(
            {
                "date": dt.datetime.fromtimestamp(timestamp, exchange_zone)
                .date()
                .isoformat(),
                "open": quote["open"][index],
                "high": quote["high"][index],
                "low": quote["low"][index],
                "close": quote["close"][index],
                "adj_close": adj_close[index],
                "volume": quote["volume"][index],
            }
        )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end-exclusive", default=DEFAULT_END_EXCLUSIVE)
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    csv_dir = args.out / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "Yahoo Finance chart API (endpoint wrapped by yfinance)",
        "paper_reference": "arXiv:2601.03802v1, Sec. III-B, footnote 1",
        "requested_start": args.start,
        "requested_end_exclusive": args.end_exclusive,
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "symbols": {},
    }
    for group, symbols in SYMBOLS.items():
        for symbol in symbols:
            url, body = fetch_chart(symbol, args.start, args.end_exclusive)
            raw_path = raw_dir / f"{symbol}.json"
            raw_path.write_bytes(body)
            rows = chart_to_rows(json.loads(body))
            csv_path = csv_dir / f"{symbol}.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest["symbols"][symbol] = {
                "group": group,
                "url": url,
                "rows": len(rows),
                "rows_with_missing_close": sum(row["close"] is None for row in rows),
                "first_date": rows[0]["date"],
                "last_date": rows[-1]["date"],
                "raw_file": str(raw_path.relative_to(args.out)),
                "raw_sha256": _sha256(raw_path),
                "csv_file": str(csv_path.relative_to(args.out)),
                "csv_sha256": _sha256(csv_path),
            }
            print(f"{symbol}: {len(rows)} rows {rows[0]['date']} -> {rows[-1]['date']}")
            time.sleep(1.0)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
