from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_RESULT_FILES = {
    "DHO": PROJECT_ROOT / "DHO" / "results" / "dho_summary.csv",
    "SEE": PROJECT_ROOT / "SEE" / "results" / "see_summary.csv",
    "DEE": PROJECT_ROOT / "DEE" / "results" / "dee_summary.csv",
    "TAF": PROJECT_ROOT / "TAF" / "results" / "cc_summary.csv",
}


def sync_selected_results() -> None:
    results_root = PROJECT_ROOT / "results"
    for group_name, source in SELECTED_RESULT_FILES.items():
        if not source.is_file():
            continue
        destination_dir = results_root / group_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)
        print(f"Synced {source} -> {destination_dir / source.name}")


if __name__ == "__main__":
    sync_selected_results()
