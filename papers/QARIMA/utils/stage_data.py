"""Pre-stage the five QARIMA datasets into ``<repo>/data/QARIMA/raw/``.

This is optional: ``lib.data.load_series`` now stages each dataset lazily (offline
for Sunspots via statsmodels; downloaded-and-cached for the rest) the first time
it is requested, so ``implementation.py`` runs end-to-end without this script.
Use it to pre-warm the cache for all five datasets at once, e.g. before running
offline. Idempotent: skips files that already exist. Run from anywhere::

    python papers/QARIMA/utils/stage_data.py [--data-root /reproduced_papers/data]

See LOG.md "Data Acquisition Log" and VISITED_URLS.md for provenance. The paper's
Sydney station 95768099999 (North Head) has no temperature data, so we stage the
substitute station 94768099999 (Sydney Observatory Hill).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import _DOWNLOAD_URLS, _ensure_downloaded, _ensure_sunspots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[3] / "data"
    ap.add_argument("--data-root", default=str(default_root))
    args = ap.parse_args()
    raw = Path(args.data_root) / "QARIMA" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Staging QARIMA data into {raw}")

    print(f"  {_ensure_sunspots(raw).name} (statsmodels, offline)")
    for name, url in _DOWNLOAD_URLS.items():
        print(f"  {name} <- {url}")
        _ensure_downloaded(raw, name)
    print("Done. NOTE: Sydney uses substitute station 94768099999 (Observatory Hill).")


if __name__ == "__main__":
    main()
