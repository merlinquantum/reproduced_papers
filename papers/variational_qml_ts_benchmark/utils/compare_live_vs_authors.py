"""Contextualise the reduced live sweep against the authors' grid-optimised results.

For the two Hénon problems run live, print (and save) side-by-side median test
MSE from:
  * our reduced live sweep (1 config/model, fixed 400-epoch budget), and
  * the authors' released grid-search best (best hyperparameter + sequence length
    per model, trained to convergence).

The comparison makes explicit WHY the reduced sweep must not be read as the
paper's result: on the hard k=4 task the reduced classical baselines are
under-parameterised/under-trained (our LSTM ~6e-2 vs the authors' grid-best
~1.6e-4), so they look artificially weak. The grid-optimised CSV reproduction
(plot_paper_figures.py) is the authoritative comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC = PROJECT_DIR / "original_results"
OUT = PROJECT_DIR / "results"
QUANTUM = {"d-QNN", "ru-QNN", "QRNN", "QLSTM", "le-QLSTM"}


def authors_best(pred: int) -> dict[str, float]:
    vqc = pd.read_csv(SRC / "vqc_averaged_ids.csv")
    F = {
        "d-QNN": vqc[vqc.Ansatz.str.startswith("paper_rivera")],
        "ru-QNN": vqc[vqc.Ansatz.str.startswith("ruexp_")],
        "QRNN": pd.read_csv(SRC / "qrnn_paper_averaged_ids.csv").query(
            "Ansatz=='paper_no_reset'"
        ),
        "QLSTM": pd.read_csv(SRC / "qlstm_paper_averaged_ids.csv"),
        "le-QLSTM": pd.read_csv(SRC / "qlstm_linear_enhanced_paper_averaged_ids.csv"),
        "MLP": pd.read_csv(SRC / "mlp_averaged_ids.csv"),
        "RNN": pd.read_csv(SRC / "rnn_averaged_ids.csv"),
        "LSTM": pd.read_csv(SRC / "lstm_averaged_ids.csv"),
    }
    out = {}
    for m, df in F.items():
        d = df[(df.Data == "henon_1000") & (df["Prediction Step"] == pred)]
        if len(d):
            b = d.loc[d["MSE Validation Median"].idxmin()]
            out[m] = (float(b["MSE Testing Median"]), int(b["Num Parameters"]))
    return out


def live_best(pred: int) -> dict[str, float]:
    df = pd.read_csv(OUT / "sweep_summary.csv")
    d = df[(df.data_label == "henon_1000") & (df.prediction_step == pred)]
    med = d.groupby("display")["mse_test"].median()
    par = d.groupby("display")["num_parameters"].first()
    return {m: (float(med[m]), int(par[m])) for m in med.index}


def main():
    lines = [
        "# Reduced live sweep vs authors' grid-optimised results (Hénon)\n",
        "Median test MSE. **Live** = our reimplementation, one representative "
        "hyperparameter per model, fixed 400-epoch budget, 3 seeds (bug-fixed). "
        "**Authors** = released grid-search best (best hyperparameter + sequence "
        "length, trained to convergence, 10 seeds).\n",
    ]
    for pred in (1, 4):
        auth = authors_best(pred)
        live = live_best(pred)
        rows = []
        for m in ["LSTM", "RNN", "MLP", "d-QNN", "ru-QNN", "QRNN", "QLSTM", "le-QLSTM"]:
            lv, lp = live.get(m, (float("nan"), 0))
            av, ap = auth.get(m, (float("nan"), 0))
            rows.append(
                {
                    "model": m,
                    "kind": "Q" if m in QUANTUM else "C",
                    "live_mse": lv,
                    "live_params": lp,
                    "authors_mse": av,
                    "authors_params": ap,
                }
            )
        tbl = pd.DataFrame(rows)
        lines.append(f"\n## Hénon k={pred}\n")
        lines.append(tbl.to_markdown(index=False, floatfmt=".3e"))
    note = (
        "\n\n**Reading this:** On k=1 (easy) our live sweep agrees with the "
        "authors that le-QLSTM/d-QNN are competitive with or slightly better than "
        "LSTM — the paper itself flags Hénon as its most quantum-favourable "
        "dataset (App. G). On k=4 (hard) the authors' grid-best classical LSTM/RNN "
        "(large hidden sizes, trained to convergence) clearly win (~1.6e-4), but "
        "our reduced fixed-budget classical baselines are under-parameterised and "
        "under-trained (~6e-2), so the live k=4 ordering is a reduced-compute "
        "artifact and must NOT be read as the paper's result. The authoritative, "
        "grid-optimised comparison across all 27 tasks is reproduced exactly in "
        "`claim_summary.md` / `ranking_all_models.png`."
    )
    lines.append(note)
    (OUT / "live_vs_authors.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
