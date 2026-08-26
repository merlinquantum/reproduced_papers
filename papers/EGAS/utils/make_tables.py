"""Aggregate egas_eval / photonic / wasserstein metrics.json into markdown tables.

Usage: python utils/make_tables.py outdir/PW/run_*/metrics.json outdir/WQ/run_*/metrics.json ...
Prints markdown to stdout (paste into README results section).
"""

from __future__ import annotations

import json
import sys
from glob import glob

PAPER_W1 = {
    "PW": 5.2380,
    "WDGV1": 5.1570,
    "DB": 13.9108,
    "WC": 10.8562,
    "WQ": 3.0112,
    "MGT": 3.3036,
    "EGSSD": 3.5619,
}


def _best(group):
    return max(g["mean_acc"] for g in group)


def main(paths):
    metrics = []
    for p in paths:
        for f in glob(p):
            metrics.append(json.load(open(f)))

    egas = [m for m in metrics if m.get("task") == "egas_eval"]
    phot = [m for m in metrics if m.get("task") == "photonic_eval"]
    was = [m for m in metrics if m.get("task") == "wasserstein"]

    if was:
        r = was[-1]["results"]
        print("### Table I — input-space 1-Wasserstein distance\n")
        print("| Dataset | Reproduced W1 | Paper W1 |")
        print("|---|---:|---:|")
        for n, v in r.items():
            if "w1" in v:
                print(f"| {n} | {v['w1']:.4f} | {PAPER_W1.get(n, float('nan')):.4f} |")
        print()

    if egas:
        print("### EGAS QKSVM accuracy (mean over splits) vs baselines\n")
        print(
            "| Dataset | W1 | best G | best G(bias) | best B(bias) | NQE | ZZ | "
            "Classical-lin | Classical-rbf | IQR | EGAS minE |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in sorted(egas, key=lambda z: z["w1"]):
            b = m["baselines"]
            print(
                f"| {m['dataset']} | {m['w1']:.2f} | {_best(m['G']):.3f} | "
                f"{_best(m['G_bias']):.3f} | {_best(m['B_bias']):.3f} | "
                f"{b['NQE']['mean_acc']:.3f} | {b['ZZ']['mean_acc']:.3f} | "
                f"{b['classical_linear']['mean_acc']:.3f} | {b['classical_rbf']['mean_acc']:.3f} | "
                f"{m['embedding_sensitivity_IQR']:.3f} | {m['egas_min_energy']:.3f} |"
            )
        print()
        print("### Win/Tie/Loss of best G(bias) vs classical linear SVM\n")
        print("| Dataset | best-G(bias) W/T/L | ZZ W/T/L | NQE W/T/L |")
        print("|---|---|---|---|")
        for m in sorted(egas, key=lambda z: z["w1"]):
            gb = m["G_bias"][m["G_bias_star_idx"]]["wtl_vs_linear"]
            zz = m["baselines"]["ZZ"]["wtl_vs_linear"]
            nqe = m["baselines"]["NQE"]["wtl_vs_linear"]

            def format_wtl(d):
                return f"{d['win']}/{d['tie']}/{d['lose']}"

            print(
                f"| {m['dataset']} | {format_wtl(gb)} | {format_wtl(zz)} | {format_wtl(nqe)} |"
            )
        print()
        print("### Bias-refinement surrogate-energy reduction ΔE (Fig 3/4)\n")
        print("| Dataset | mean ΔE (G group) | mean ΔE (B group) |")
        print("|---|---:|---:|")
        import statistics as st

        for m in sorted(egas, key=lambda z: z["w1"]):
            dg = m["delta_E"]["G"]
            db = m["delta_E"]["B"]
            print(f"| {m['dataset']} | {st.mean(dg):+.4f} | {st.mean(db):+.4f} |")
        print()

    if phot:
        print("### MerLin photonic counterpart (fidelity-kernel QKSVM)\n")
        h = phot[0]["hardware"]
        print(
            f"Hardware: {h['n_photons']} photons, {h['n_modes']} modes, {h['computation_space']}, "
            f"{h['detector']} detectors, {h['simulator']}.\n"
        )
        print(
            "| Dataset | W1 | photonic-fixed | photonic-trained | ZZ | Classical-lin |"
        )
        print("|---|---:|---:|---:|---:|---:|")
        for m in sorted(phot, key=lambda z: z["w1"]):
            fixed = m.get("photonic_fixed")
            trained = m.get("photonic_trained")
            if fixed is None:
                fixed_mean = _best(m["G"])
            else:
                fixed_mean = fixed["mean_acc"]
            if trained is None:
                trained_mean = _best(m["G_bias"])
            else:
                trained_mean = trained["mean_acc"]
            print(
                f"| {m['dataset']} | {m['w1']:.2f} | {fixed_mean:.3f} | "
                f"{trained_mean:.3f} | {m['ZZ']['mean_acc']:.3f} | "
                f"{m['classical_linear']['mean_acc']:.3f} |"
            )
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
