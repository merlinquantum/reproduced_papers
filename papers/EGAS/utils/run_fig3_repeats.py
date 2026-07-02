"""Faithful Fig 3: per-candidate bias-refinement ΔE with error bars over repeats (PW).

The paper refines each of the 10 lowest-energy (G) and 10 highest-energy (B) EGAS candidates
multiple times and plots mean ± std ΔE. The standard egas_eval run only does one refinement per
candidate; this script re-runs EGAS on PW, selects top-10/bottom-10, refines each `--repeats`
times, and saves `outdir/fig3_PW/fig3_data.json` (consumed by `plot_paper_figures.fig3`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bias import refine_bias  # noqa: E402
from lib.circuits import build_token_pool  # noqa: E402
from lib.data import load_dataset, make_slices  # noqa: E402
from lib.egas import run_egas, unique_sorted_candidates  # noqa: E402

DATASET = "PW"
N_QUBITS = 8
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def main():
    pool = build_token_pool(N_QUBITS)
    X, y = load_dataset(
        DATASET,
        data_root=str(ROOT.parent.parent / "data"),
        n_components=N_QUBITS,
        seed=0,
    )
    slices = make_slices(X, y, n_train=400, n_test=50, n_repeats=1, seed=0)
    rng = np.random.default_rng(0)
    Xtr0 = slices[0]["X_train"]
    idx = rng.choice(len(Xtr0), 36, replace=False)
    Xe, ye = Xtr0[idx], slices[0]["y_train"][idx]

    print(f"[fig3] EGAS search on {DATASET}...")
    _, _, buf = run_egas(
        pool,
        Xe,
        ye,
        N_QUBITS,
        seq_len=28,
        n_iters=120,
        n_candidates=12,
        select_k=6,
        gamma=0.1,
        d_model=32,
        n_layers=1,
        n_heads=2,
        seed=0,
    )
    G_ids, B_ids = unique_sorted_candidates(buf, top=TOPK, bottom=TOPK)

    def deltas(ids_list, tag):
        means, stds = [], []
        for k, sid in enumerate(ids_list):
            seq = [pool[i] for i in sid]
            dEs = []
            for r in range(REPEATS):
                _, Eb, Ea = refine_bias(
                    seq, Xe, ye, N_QUBITS, epochs=120, batch_samples=25, lr=5e-4, seed=r
                )
                dEs.append(Eb - Ea)
            means.append(float(np.mean(dEs)))
            stds.append(float(np.std(dEs)))
            print(f"[fig3] {tag}{k + 1}: ΔE={means[-1]:+.4f} ± {stds[-1]:.4f}")
        return means, stds

    Gm, Gs = deltas(G_ids, "G")
    Bm, Bs = deltas(B_ids, "B")
    out = ROOT / "outdir" / "fig3_PW"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "dataset": DATASET,
            "repeats": REPEATS,
            "G_mean": Gm,
            "G_std": Gs,
            "B_mean": Bm,
            "B_std": Bs,
        },
        open(out / "fig3_data.json", "w"),
        indent=2,
    )
    print("[fig3] wrote", out / "fig3_data.json")


if __name__ == "__main__":
    main()
