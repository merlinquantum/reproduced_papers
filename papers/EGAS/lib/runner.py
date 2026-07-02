"""Runtime entry point for the EGAS reproduction.

Dispatches on ``cfg["task"]``:
  * ``wasserstein`` — input-space 1-Wasserstein distances per dataset (Table I).
                      Supports binary and multiclass (returns mean pairwise W1).
  * ``fig1``        — trace distance vs W1 for a ZZ feature map (Fig. 1).
  * ``egas_eval``   — full pipeline on one dataset: EGAS search, bias refinement of the
                      G/B groups, QKSVM evaluation over splits, and ZZ / NQE / classical
                      baselines (Figs. 3-7). Supports binary and multiclass labels.
  * ``photonic_eval`` — photonic EGAS search, encoder-parameter refinement, photonic QKSVM
                        evaluation over splits, and classical / gate baselines.
                        Supports binary and multiclass labels.
All runs write ``metrics.json`` (+ task-specific NPZ) into ``run_dir``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch


def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    task = cfg.get("task", "egas_eval")
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))
    if task == "wasserstein":
        _run_wasserstein(cfg, run_dir, logger)
    elif task == "fig1":
        _run_fig1(cfg, run_dir, logger)
    elif task == "egas_eval":
        _run_egas_eval(cfg, run_dir, logger)
    elif task in {"photonic_eval", "photonic_egas_eval"}:
        _run_photonic_eval(cfg, run_dir, logger)
    else:
        raise ValueError(f"unknown task {task}")


def _run_photonic_eval(cfg, run_dir, logger):
    """Photonic EGAS search/refinement + photonic QKSVM evaluation.
    
    Supports binary and multiclass labels. For multiclass, pairwise energy uses
    same-class matching and classical baselines use one-vs-rest (OvR) strategy.
    """
    from .data import load_dataset, make_slices
    from .kernel_svm import nqe_accuracy, zz_accuracy
    from .photonic_circuits import build_token_pool, create_quantum_module
    from .photonic_egas import (
        refine_candidates,
        run_egas,
        unique_sorted_candidates,
    )
    from .photonic_kernel_svm import (
        classical_svm_accuracy,
    )
    from .photonic_kernel_svm import (
        qksvm_accuracy as photonic_qksvm_accuracy,
    )
    from .wasserstein import dataset_wasserstein

    seed = int(cfg.get("seed", 0))
    device = cfg.get("device", "cpu")
    dcfg = cfg["dataset"]
    pcfg = cfg.get("photonic", {})
    ecfg = cfg.get("egas", {})
    bcfg = cfg.get("bias", {})
    vcfg = cfg["eval"]
    name = dcfg["name"]
    n_modes = int(dcfg.get("n_qubits", 8))
    n_photons = int(pcfg.get("n_photons", 2))
    computation_space = pcfg.get("computation_space", "UNBUNCHED")
    if isinstance(computation_space, str):
        computation_space = computation_space.upper()
    X, y = load_dataset(name, data_root=dcfg["root"], n_components=n_modes, seed=seed)
    w1 = dataset_wasserstein(X, y, seed=seed)
    slices = make_slices(
        X,
        y,
        n_train=vcfg["n_train"],
        n_test=vcfg["n_test"],
        n_repeats=vcfg["n_repeats"],
        seed=seed,
    )

    pool = build_token_pool(n_modes)

    # Photonic EGAS search on a fixed sample batch drawn from the first split's train set.
    rng = np.random.default_rng(seed)
    Xtr0 = slices[0]["X_train"]
    search_samples = int(ecfg.get("search_samples", pcfg.get("search_samples", 24)))
    s = min(search_samples, len(Xtr0))
    idx = rng.choice(len(Xtr0), s, replace=False)
    Xe, ye = Xtr0[idx], slices[0]["y_train"][idx]

    t0 = time.time()
    gpt, hist, buf = run_egas(
        pool,
        Xe,
        ye,
        n_modes,
        seq_len=int(ecfg.get("seq_len", pcfg.get("seq_len", 28))),
        num_photons=n_photons,
        computation_space=computation_space,
        n_iters=int(ecfg.get("n_iters", pcfg.get("n_iters", 20))),
        n_candidates=int(ecfg.get("n_candidates", pcfg.get("n_candidates", 8))),
        select_k=int(ecfg.get("select_k", pcfg.get("select_k", 4))),
        gamma=float(ecfg.get("gamma", pcfg.get("gamma", 0.1))),
        lr=float(ecfg.get("lr", pcfg.get("egas_lr", 5e-5))),
        temp_max=float(ecfg.get("temp_max", pcfg.get("temp_max", 100.0))),
        temp_min=float(ecfg.get("temp_min", pcfg.get("temp_min", 0.04))),
        d_model=int(ecfg.get("d_model", pcfg.get("d_model", 32))),
        n_layers=int(ecfg.get("n_layers", pcfg.get("gpt_layers", 1))),
        n_heads=int(ecfg.get("n_heads", pcfg.get("n_heads", 2))),
        seed=seed,
        device=device,
        logger=logger,
    )
    search_time = time.time() - t0
    top = int(vcfg.get("top", 4))
    G_ids, B_ids = unique_sorted_candidates(buf, top=top, bottom=top)

    def build_unrefined(ids_list):
        group = []
        for sid in ids_list:
            seq = [pool[int(i)] for i in sid]
            encoder = create_quantum_module(
                seq,
                n_modes=n_modes,
                num_photons=n_photons,
                computation_space=computation_space,
            )
            group.append({"seq": seq, "encoder": encoder})
        return group

    G = build_unrefined(G_ids)
    B = build_unrefined(B_ids)
    refine_common = {
        "epochs": int(bcfg.get("epochs", pcfg.get("epochs", 25))),
        "batch_samples": int(bcfg.get("batch_samples", pcfg.get("batch", 24))),
        "lr": float(bcfg.get("lr", pcfg.get("lr", 5e-4))),
        "seed": seed,
        "computation_space": computation_space,
    }
    G_refined = refine_candidates(
        G_ids,
        pool,
        Xe,
        ye,
        n_modes,
        num_photons=n_photons,
        device=device,
        **refine_common,
    )
    B_refined = refine_candidates(
        B_ids,
        pool,
        Xe,
        ye,
        n_modes,
        num_photons=n_photons,
        device=device,
        **refine_common,
    )

    def eval_group(group):
        accs = np.zeros((len(group), len(slices)))
        for k, item in enumerate(group):
            for j, sl in enumerate(slices):
                accs[k, j] = photonic_qksvm_accuracy(
                    item["encoder"],
                    sl["X_train"],
                    sl["y_train"],
                    sl["X_test"],
                    sl["y_test"],
                    device=device,
                )
        return accs

    accG = eval_group(G)
    accGb = eval_group(G_refined)
    accB = eval_group(B)
    accBb = eval_group(B_refined)

    n_sp = len(slices)
    zz = np.array(
        [
            zz_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], n_modes
            )
            for sl in slices
        ]
    )
    lin = np.array(
        [
            classical_svm_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], "linear"
            )
            for sl in slices
        ]
    )
    rbf = np.array(
        [
            classical_svm_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], "rbf"
            )
            for sl in slices
        ]
    )
    nqe_epochs = cfg.get("nqe", {}).get("epochs")
    if nqe_epochs is None:
        nqe = np.full(n_sp, np.nan)
    else:
        nqe = np.array(
            [
                nqe_accuracy(
                    sl["X_train"],
                    sl["y_train"],
                    sl["X_test"],
                    sl["y_test"],
                    n_modes,
                    epochs=nqe_epochs,
                    seed=seed,
                    device=device,
                )
                for sl in slices
            ]
        )

    for j in range(n_sp):
        logger.info(
            "[photonic-egas %s] split %d: G=%.3f G_refined=%.3f B=%.3f B_refined=%.3f ZZ=%.3f lin=%.3f",
            name,
            j,
            float(accG[:, j].max()) if len(G) else float("nan"),
            float(accGb[:, j].max()) if len(G_refined) else float("nan"),
            float(accB[:, j].max()) if len(B) else float("nan"),
            float(accBb[:, j].max()) if len(B_refined) else float("nan"),
            zz[j],
            lin[j],
        )

    def stat(a):
        a = np.array(a)
        return {"mean_acc": float(np.nanmean(a)), "std_acc": float(np.nanstd(a))}

    def wtl(acc_row):
        w = int((acc_row > lin).sum())
        num_losses = int((acc_row < lin).sum())
        return {"win": w, "tie": n_sp - w - num_losses, "lose": num_losses}

    def best_rep(accs):
        if len(accs) == 0:
            return None
        wins = (accs > lin[None, :]).sum(axis=1)
        return int(np.argmax(wins))

    def summarize(accs, group):
        return [
            {
                "E_before": group[k].get("E_before"),
                "E_after": group[k].get("E_after"),
                "mean_acc": float(accs[k].mean()),
                "std_acc": float(accs[k].std()),
                "wtl_vs_linear": wtl(accs[k]),
            }
            for k in range(len(group))
        ]

    mean_acc_parts = [
        accG.mean(1),
        accGb.mean(1),
        accB.mean(1),
        accBb.mean(1),
        [zz.mean()],
    ]
    if not np.isnan(nqe).all():
        mean_acc_parts.append([np.nanmean(nqe)])
    mean_accs = np.concatenate(mean_acc_parts)
    iqr = float(np.percentile(mean_accs, 75) - np.percentile(mean_accs, 25))

    metrics = {
        "task": "photonic_eval",
        "dataset": name,
        "w1": w1,
        "hardware": {
            "computation_space": computation_space,
            "detector": "threshold",
            "n_photons": n_photons,
            "n_modes": n_modes,
            "encoding": "EGAS photonic token circuit (PS/BS)",
            "measurement": "QuantumLayer amplitudes + fidelity kernel",
            "postselection": "none",
            "simulator": "MerLin SLOS analytic (shots=None)",
            "n_layers": None,
        },
        "n_modes": n_modes,
        "search_time_sec": search_time,
        "n_splits": n_sp,
        "egas_min_energy": float(min(b[1] for b in buf)),
        "buffer_size": len(buf),
        "baselines": {
            "classical_linear": stat(lin),
            "classical_rbf": stat(rbf),
            "ZZ": {**stat(zz), "wtl_vs_linear": wtl(zz)},
            "NQE": (
                {**stat(nqe), "wtl_vs_linear": wtl(nqe)}
                if not np.isnan(nqe).all()
                else None
            ),
        },
        "G": summarize(accG, G),
        "G_bias": summarize(accGb, G_refined),
        "B": summarize(accB, B),
        "B_bias": summarize(accBb, B_refined),
        "G_star_idx": best_rep(accG),
        "G_bias_star_idx": best_rep(accGb),
        "B_star_idx": best_rep(accB),
        "B_bias_star_idx": best_rep(accBb),
        "embedding_sensitivity_IQR": iqr,
        "delta_E": {
            "G": [g["E_before"] - g["E_after"] for g in G_refined],
            "B": [b["E_before"] - b["E_after"] for b in B_refined],
        },
    }
    _save_json(run_dir / "metrics.json", metrics)
    np.savez(
        run_dir / "photonic_acc.npz",
        accG=accG,
        accGb=accGb,
        accB=accB,
        accBb=accBb,
        zz=zz,
        lin=lin,
        rbf=rbf,
        nqe=nqe,
    )
    best_gb = max((g["mean_acc"] for g in metrics["G_bias"]), default=float("nan"))
    logger.info(
        "[photonic-egas %s] W1=%.3f | best G(refined) mean acc=%.3f | ZZ=%.3f Lin=%.3f | IQR=%.3f",
        name,
        w1,
        best_gb,
        float(zz.mean()),
        float(lin.mean()),
        iqr,
    )


def _run_wasserstein(cfg, run_dir, logger):
    from .data import DATASETS, load_dataset
    from .wasserstein import dataset_wasserstein

    data_root = cfg["dataset"]["root"]
    names = cfg.get("datasets") or list(DATASETS.keys())
    seed = int(cfg.get("seed", 0))
    out = {}
    for name in names:
        try:
            X, y = load_dataset(name, data_root=data_root, seed=seed)
            w1 = dataset_wasserstein(X, y, seed=seed)
            out[name] = {
                "w1": w1,
                "n": int(len(X)),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == -1).sum()),
            }
            logger.info("W1 %s = %.4f (n=%d)", name, w1, len(X))
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            logger.warning("W1 %s failed: %s", name, e)
    _save_json(run_dir / "metrics.json", {"task": "wasserstein", "results": out})


def _run_fig1(cfg, run_dir, logger):
    from .wasserstein import fig1_curve

    fc = cfg.get("fig1", {})
    out = {}
    for L in fc.get("layers", [1, 2]):
        w1s, tds = fig1_curve(
            n_qubits=fc.get("n_qubits", 4),
            n_layers=L,
            n_per_class=fc.get("n_per_class", 60),
            seed=int(cfg.get("seed", 0)),
        )
        out[f"L{L}"] = {"w1": w1s.tolist(), "trace_dist": tds.tolist()}
        logger.info(
            "Fig1 L=%d: W1 %.2f-%.2f, trace_dist %.3f-%.3f",
            L,
            w1s.min(),
            w1s.max(),
            tds.min(),
            tds.max(),
        )
    _save_json(run_dir / "metrics.json", {"task": "fig1", "results": out})


def _run_egas_eval(cfg, run_dir, logger):
    """Gate-based EGAS search/refinement + gate QKSVM evaluation.
    
    Supports binary and multiclass labels. For multiclass, pairwise energy uses
    same-class matching and classical baselines use one-vs-rest (OvR) strategy.
    """
    from .bias import refine_bias
    from .circuits import build_token_pool
    from .data import load_dataset, make_slices
    from .egas import run_egas, unique_sorted_candidates
    from .kernel_svm import (
        classical_svm_accuracy,
        nqe_accuracy,
        qksvm_accuracy,
        zz_accuracy,
    )
    from .wasserstein import dataset_wasserstein

    seed = int(cfg.get("seed", 0))
    dcfg, ecfg, bcfg, vcfg = cfg["dataset"], cfg["egas"], cfg["bias"], cfg["eval"]
    name = dcfg["name"]
    n_qubits = int(dcfg.get("n_qubits", 8))
    pool = build_token_pool(n_qubits)
    X, y = load_dataset(name, data_root=dcfg["root"], n_components=n_qubits, seed=seed)
    w1 = dataset_wasserstein(X, y, seed=seed)
    slices = make_slices(
        X,
        y,
        n_train=vcfg["n_train"],
        n_test=vcfg["n_test"],
        n_repeats=vcfg["n_repeats"],
        seed=seed,
    )

    # EGAS search on a fixed sample batch drawn from the first split's train set.
    rng = np.random.default_rng(seed)
    Xtr0 = slices[0]["X_train"]
    s = min(ecfg["search_samples"], len(Xtr0))
    idx = rng.choice(len(Xtr0), s, replace=False)
    Xe, ye = Xtr0[idx], slices[0]["y_train"][idx]
    t0 = time.time()
    gpt, hist, buf = run_egas(
        pool,
        Xe,
        ye,
        n_qubits,
        seq_len=ecfg["seq_len"],
        n_iters=ecfg["n_iters"],
        n_candidates=ecfg["n_candidates"],
        select_k=ecfg["select_k"],
        gamma=ecfg["gamma"],
        lr=ecfg.get("lr", 5e-5),
        temp_max=ecfg.get("temp_max", 100.0),
        temp_min=ecfg.get("temp_min", 0.04),
        d_model=ecfg.get("d_model", 64),
        n_layers=ecfg.get("n_layers", 2),
        n_heads=ecfg.get("n_heads", 4),
        seed=seed,
        logger=logger,
    )
    search_time = time.time() - t0
    top = vcfg.get("top", 5)
    G_ids, B_ids = unique_sorted_candidates(buf, top=top, bottom=top)

    # Bias-refine each selected candidate.
    def refine_all(ids_list):
        out = []
        for sid in ids_list:
            seq = [pool[i] for i in sid]
            bias, Eb, Ea = refine_bias(
                seq,
                Xe,
                ye,
                n_qubits,
                epochs=bcfg["epochs"],
                batch_samples=bcfg.get("batch_samples", 25),
                lr=bcfg.get("lr", 5e-4),
                seed=seed,
            )
            out.append({"seq": seq, "bias": bias, "E_before": Eb, "E_after": Ea})
        return out

    G = refine_all(G_ids)
    B = refine_all(B_ids)

    def eval_group(group, use_bias):
        accs = np.zeros((len(group), len(slices)))
        for k, item in enumerate(group):
            for j, sl in enumerate(slices):
                accs[k, j] = qksvm_accuracy(
                    item["seq"],
                    sl["X_train"],
                    sl["y_train"],
                    sl["X_test"],
                    sl["y_test"],
                    n_qubits,
                    bias=item["bias"] if use_bias else None,
                )
        return accs

    accG = eval_group(G, False)
    accGb = eval_group(G, True)
    accB = eval_group(B, False)
    accBb = eval_group(B, True)

    n_sp = len(slices)
    zz = np.array(
        [
            zz_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], n_qubits
            )
            for sl in slices
        ]
    )
    lin = np.array(
        [
            classical_svm_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], "linear"
            )
            for sl in slices
        ]
    )
    rbf = np.array(
        [
            classical_svm_accuracy(
                sl["X_train"], sl["y_train"], sl["X_test"], sl["y_test"], "rbf"
            )
            for sl in slices
        ]
    )
    nqe = np.array(
        [
            nqe_accuracy(
                sl["X_train"],
                sl["y_train"],
                sl["X_test"],
                sl["y_test"],
                n_qubits,
                epochs=cfg.get("nqe", {}).get("epochs", 80),
                seed=seed,
            )
            for sl in slices
        ]
    )

    def wtl(acc_row):  # win/tie/loss vs linear baseline across splits
        w = int((acc_row > lin).sum())
        num_losses = int((acc_row < lin).sum())
        return {"win": w, "tie": n_sp - w - num_losses, "lose": num_losses}

    def best_rep(accs):  # representative = embedding with most wins vs linear baseline
        wins = (accs > lin[None, :]).sum(axis=1)
        return int(np.argmax(wins))

    def summarize(accs, group):
        return [
            {
                "E_before": group[k]["E_before"],
                "E_after": group[k]["E_after"],
                "mean_acc": float(accs[k].mean()),
                "std_acc": float(accs[k].std()),
                "wtl_vs_linear": wtl(accs[k]),
            }
            for k in range(len(group))
        ]

    # embedding-sensitivity IQR (Fig 6): IQR of mean test acc over quantum embeddings.
    mean_accs = np.concatenate(
        [
            accG.mean(1),
            accGb.mean(1),
            accB.mean(1),
            accBb.mean(1),
            [zz.mean(), nqe.mean()],
        ]
    )
    iqr = float(np.percentile(mean_accs, 75) - np.percentile(mean_accs, 25))

    metrics = {
        "task": "egas_eval",
        "dataset": name,
        "w1": w1,
        "n_qubits": n_qubits,
        "search_time_sec": search_time,
        "n_splits": n_sp,
        "egas_min_energy": float(min(b[1] for b in buf)),
        "buffer_size": len(buf),
        "baselines": {
            "classical_linear": {
                "mean_acc": float(lin.mean()),
                "std_acc": float(lin.std()),
            },
            "classical_rbf": {
                "mean_acc": float(rbf.mean()),
                "std_acc": float(rbf.std()),
            },
            "ZZ": {
                "mean_acc": float(zz.mean()),
                "std_acc": float(zz.std()),
                "wtl_vs_linear": wtl(zz),
            },
            "NQE": {
                "mean_acc": float(nqe.mean()),
                "std_acc": float(nqe.std()),
                "wtl_vs_linear": wtl(nqe),
            },
        },
        "G": summarize(accG, G),
        "G_bias": summarize(accGb, G),
        "B": summarize(accB, B),
        "B_bias": summarize(accBb, B),
        "G_star_idx": best_rep(accG),
        "G_bias_star_idx": best_rep(accGb),
        "B_star_idx": best_rep(accB),
        "B_bias_star_idx": best_rep(accBb),
        "embedding_sensitivity_IQR": iqr,
        "delta_E": {
            "G": [g["E_before"] - g["E_after"] for g in G],
            "B": [b["E_before"] - b["E_after"] for b in B],
        },
    }
    _save_json(run_dir / "metrics.json", metrics)
    np.savez(
        run_dir / "accuracies.npz",
        accG=accG,
        accGb=accGb,
        accB=accB,
        accBb=accBb,
        zz=zz,
        lin=lin,
        rbf=rbf,
        nqe=nqe,
    )
    logger.info(
        "[%s] W1=%.3f | best G(bias) mean acc=%.3f | ZZ=%.3f NQE=%.3f Lin=%.3f | IQR=%.3f",
        name,
        w1,
        max(g["mean_acc"] for g in metrics["G_bias"]),
        zz.mean(),
        nqe.mean(),
        lin.mean(),
        iqr,
    )
