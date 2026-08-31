"""Stage the five QARIMA datasets into ``<repo>/data/QARIMA/raw/``.

Idempotent: skips files that already exist. Sunspots + CO2 need no download beyond
statsmodels / a small Rdatasets CSV; AusBeer, Woolyarn and the Sydney NOAA feed are
fetched once. Run from anywhere::

    python papers/QARIMA/utils/stage_data.py [--data-root /reproduced_papers/data]

See LOG.md "Data Acquisition Log" and VISITED_URLS.md for provenance. The paper's
Sydney station 95768099999 (North Head) has no temperature data, so we stage the
substitute station 94768099999 (Sydney Observatory Hill).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"
_NOAA = "https://www.ncei.noaa.gov/data/global-hourly/access/2024"
_FILES = {
    "ausbeer.csv": f"{_RDATASETS}/fpp2/ausbeer.csv",
    "woolyrnq.csv": f"{_RDATASETS}/forecast/woolyrnq.csv",
    "co2_R_datasets.csv": f"{_RDATASETS}/timeSeriesDataSets/co2_ts.csv",
    "sydney_observatory_hill_2024.csv": f"{_NOAA}/94768099999.csv",
}


def _get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {dest.name} (exists)")
        return
    print(f"  download {dest.name} <- {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def main() -> None:
    ap = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[3] / "data"
    ap.add_argument("--data-root", default=str(default_root))
    args = ap.parse_args()
    raw = Path(args.data_root) / "QARIMA" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Staging QARIMA data into {raw}")

    # Sunspots via statsmodels (bundled, offline)
    ss = raw / "sunspots.csv"
    if not ss.exists():
        import statsmodels.api as sm

        sm.datasets.sunspots.load_pandas().data.to_csv(ss, index=False)
        print(f"  wrote {ss.name} (statsmodels, offline)")
    else:
        print(f"  skip {ss.name} (exists)")

    for name, url in _FILES.items():
        _get(url, raw / name)
    print("Done. NOTE: Sydney uses substitute station 94768099999 (Observatory Hill).")


if __name__ == "__main__":
    main()
