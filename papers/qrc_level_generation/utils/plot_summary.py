"""Generate the headline reproduction figure.

Combines:
- Markov / uncorrelated baselines (computed once),
- the published Aer T=1 reference,
- our trained gate-based QRC at T=1,
- our trained photonic MerLin QRC at T=1.

Each curve plots originality vs. sequence length L. Outputs:
    results/originality_combined.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _PAPER_DIR / "results"

if str(_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_DIR))


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_xy(originality: dict[str, float]) -> tuple[list[int], list[float]]:
    lengths = sorted(int(k) for k in originality.keys())
    values = [float(originality[str(L)]) for L in lengths]
    return lengths, values


def main(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = _load_metrics(_RESULTS_DIR / "reference_eval_metrics.json")
    qubit = _load_metrics(_RESULTS_DIR / "qrc_qubit_metrics.json")
    phot = _load_metrics(_RESULTS_DIR / "qrc_photonic_metrics.json")

    fig, ax = plt.subplots(figsize=(7, 4))

    xs, ys = _to_xy(ref["baselines"]["markov"]["originality"])
    ax.plot(xs, ys, "k--", label="Markov baseline")
    xs, ys = _to_xy(ref["baselines"]["uncorrelated"]["originality"])
    ax.plot(xs, ys, "k-.", label="Uncorrelated baseline")

    if "Aer_T=1.0" in ref.get("reference", {}):
        xs, ys = _to_xy(ref["reference"]["Aer_T=1.0"]["originality"])
        ax.plot(xs, ys, "g^-", label="Paper QRC Aer T=1 (Moth open data)")

    if "QRC_T=1.0" in qubit.get("qrc", {}):
        xs, ys = _to_xy(qubit["qrc"]["QRC_T=1.0"]["originality"])
        ax.plot(xs, ys, "C0o-", label="Reproduction gate QRC T=1")

    if "QRC_T=1.0" in phot.get("qrc", {}):
        xs, ys = _to_xy(phot["qrc"]["QRC_T=1.0"]["originality"])
        ax.plot(xs, ys, "C1s-", label="MerLin photonic QRC T=1")

    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Originality rate")
    ax.set_title("QRC level generation - originality vs. L")
    ax.set_xticks(list(range(2, 21, 2)))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_RESULTS_DIR / "originality_combined.png",
        help="Where to save the combined figure.",
    )
    args = parser.parse_args()
    main(args.out)
