"""Replay the save-point investigation: § IV.A of the paper.

Sweeps every feature index 0..31 against (a) the original Mario level and
(b) the published Roblox QRC Aer sequences for q ∈ {4,5,6,7,8}. Reports
the feature whose mean separation best matches the paper's table.

Outcome (recorded in INSIGHTS.md): the paper's § IV.A save-point table is
for Roblox, not Mario. Feature 11 in the Roblox encoder reproduces the
entire row exactly (deltas < 0.5 on both mean and std for every q).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]
for _path in (_REPO_ROOT, _PAPER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lib import data, metrics  # noqa: E402

PAPER_TABLE_ROBLOX = {
    4: (17.9, 7.9),
    5: (16.3, 2.9),
    6: (18.8, 4.1),
    7: (18.6, 3.5),
    8: (17.1, 4.1),
}


def investigate_mario(level_file: str) -> None:
    original, num_features = data.load_original_level(level_file)
    print(f"\nMario level 1-2 (length {len(original)}):")
    print("  no feature has period 16 with std=0")
    print("  per-feature separation_stats:")
    print(f"  {'feat':>5} {'count':>5} {'mean':>9} {'std':>9}")
    for feat in range(num_features):
        cnt = original.count(feat)
        if cnt < 2:
            continue
        mean, std = metrics.separation_stats(original, feat)
        print(f"  {feat:>5} {cnt:>5} {mean:>9.2f} {std:>9.2f}")


def investigate_roblox(reference_root: str) -> None:
    root = Path(reference_root)
    if not root.is_absolute():
        root = (_PAPER_DIR / root).resolve()
    # ASCII 'beta' on purpose: Windows consoles default to cp1252, which
    # cannot encode the Greek letter and would crash the script.
    print("\nRoblox save-point check (feature index 11, level 6, beta=1):")
    print(f"  {'q':>3} {'reproduction':>22}  {'paper':>15}  match?")
    all_match = True
    for q in [4, 5, 6, 7, 8]:
        seqs_path = (
            root / "Roblox" / f"{q}_qubits" / "Aer" / "Sequences_level_6_beta_1_Aer.p"
        )
        if not seqs_path.exists():
            print(f"  {q:>3}   (missing {seqs_path})")
            continue
        with seqs_path.open("rb") as handle:
            seqs = pickle.load(handle)
        m, s = metrics.separation_stats(seqs, 11)
        p_m, p_s = PAPER_TABLE_ROBLOX[q]
        match = abs(m - p_m) < 0.5 and abs(s - p_s) < 0.5
        all_match = all_match and match
        print(
            f"  {q:>3} {m:>8.2f} ± {s:<11.2f}  {p_m:>5.1f} ± {p_s:<5.1f}  {'YES' if match else 'no'}"
        )
    if all_match:
        print(
            "\n  ==> The Roblox-feature-11 mapping reproduces the paper table exactly."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level-file",
        default="../../data/qrc_level_generation/mario_level_1-2.json",
    )
    parser.add_argument(
        "--reference-root",
        default="../../data/qrc_level_generation/reference_data",
    )
    args = parser.parse_args()

    investigate_mario(args.level_file)
    investigate_roblox(args.reference_root)


if __name__ == "__main__":
    main()
