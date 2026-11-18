"""No-flank MHCflurry presentation screening of existing generation attempts."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import sha256, write_json
from .data_setup import AMINO_ACIDS, load_evaluation_hlas

KEYS = ["prior", "seed", "hla"]
CUTOFFS = (0.1, 0.25, 0.5, 1, 2, 5)
SCORES = [
    "affinity",
    "affinity_percentile",
    "processing_score",
    "presentation_score",
    "presentation_percentile",
]


def validate_attempts(
    frame: pd.DataFrame, hlas: list[str], attempts: int = 1000
) -> pd.DataFrame:
    """Require complete condition/seed panels; never discard invalid attempts.

    Parameters
    ----------
    frame : pandas.DataFrame
        Existing rows with prior, seed, HLA, attempt and peptide identities.
    hlas : list[str]
        Complete expected evaluation panel.
    attempts : int
        Required attempts per HLA and condition/seed.

    Returns
    -------
    pandas.DataFrame
        Copy with validity annotations and the original attempt order.
    """
    required = set(KEYS + ["attempt", "peptide"])
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError(f"Input requires nonempty columns {sorted(required)}")
    frame = frame.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing generation identity or sequence")
    if not set(frame.prior) <= {"gaussian", "dist", "indist"}:
        raise ValueError(
            "Only Gaussian and selected fixed-Fock Dist/Indist are supported"
        )
    for name in ("seed", "attempt"):
        if not np.equal(frame[name], frame[name].astype(int)).all():
            raise ValueError(f"Noninteger {name}")
    if frame.duplicated(KEYS + ["attempt"]).any():
        raise ValueError("Duplicate attempt identity")
    for _, group in frame.groupby(["prior", "seed"], sort=False):
        if set(group.hla) != set(hlas):
            raise ValueError("Missing or extra HLA in a condition/seed panel")
        for _, hla_group in group.groupby("hla", sort=False):
            if set(hla_group.attempt) != set(range(1, attempts + 1)):
                raise ValueError("Incomplete attempt numbering")
    peptide = frame.peptide
    if not peptide.map(lambda p: isinstance(p, str)).all():
        raise ValueError("Peptides must be strings")
    frame["unscored_reason"] = [
        "too_short"
        if len(p) < 8
        else "unsupported_length"
        if len(p) > 9
        else "nonstandard_residue"
        if not set(p) <= AMINO_ACIDS
        else ""
        for p in peptide
    ]
    frame["canonical_9mer"] = peptide.map(
        lambda p: len(p) == 9 and set(p) <= AMINO_ACIDS
    )
    return frame


def score_attempts(
    frame: pd.DataFrame, predictor, hlas: list[str], attempts: int = 1000
) -> pd.DataFrame:
    """Score unique valid pairs, validate identity, and expand back to every attempt.

    Parameters
    ----------
    frame : pandas.DataFrame
        Existing generation attempts, including invalid sequences.
    predictor : object
        MHCflurry-compatible presentation predictor using no flank arguments.
    hlas : list[str]
        Complete expected HLA panel.
    attempts : int
        Required attempts per panel member.

    Returns
    -------
    pandas.DataFrame
        One output row per attempt; invalid sequences have missing scores.
    """
    frame = validate_attempts(frame, hlas, attempts)
    frame = frame.drop(columns=[c for c in SCORES if c in frame])
    query = (
        frame.loc[frame.unscored_reason.eq(""), ["hla", "peptide"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    scored = query.copy()
    if len(query):
        predicted = (
            predictor.predict(
                peptides=query.peptide.tolist(),
                alleles={h: [h] for h in hlas},
                sample_names=query.hla.tolist(),
                include_affinity_percentile=True,
                verbose=0,
                throw=True,
            )
            .sort_values("peptide_num")
            .reset_index(drop=True)
        )
        if (
            predicted.peptide_num.tolist() != list(range(len(query)))
            or predicted.peptide.tolist() != query.peptide.tolist()
            or predicted.sample_name.tolist() != query.hla.tolist()
        ):
            raise ValueError("Predictor returned mismatched peptide/HLA identities")
        if not np.isfinite(predicted[SCORES].to_numpy()).all():
            raise ValueError("Nonfinite predictor scores for scorable attempts")
        if not predicted.presentation_percentile.between(0, 100).all():
            raise ValueError("Presentation percentiles outside [0, 100]")
        for column in SCORES:
            scored[column] = predicted[column].to_numpy()
    else:
        for column in SCORES:
            scored[column] = pd.Series(dtype=float)
    result = frame.merge(
        scored, on=["hla", "peptide"], how="left", sort=False, validate="many_to_one"
    )
    if not result[KEYS + ["attempt", "peptide"]].equals(
        frame[KEYS + ["attempt", "peptide"]]
    ):
        raise ValueError("Attempt order changed during score expansion")
    return result


def aggregate_attempts(
    scored: pd.DataFrame, hlas: list[str], attempts: int = 1000
) -> pd.DataFrame:
    """Use all generation attempts as denominator, with strict percentile cutoffs.

    Parameters
    ----------
    scored : pandas.DataFrame
        Attempt-level output of ``score_attempts``.
    hlas : list[str]
        Complete expected panel.
    attempts : int
        Required attempts per HLA.

    Returns
    -------
    pandas.DataFrame
        Per-condition/seed/HLA screening metrics.
    """
    validated = validate_attempts(scored, hlas, attempts)
    if not set(SCORES) <= set(scored.columns):
        raise ValueError("Missing predictor score columns")
    valid = validated.unscored_reason.eq("")
    if not np.isfinite(scored.loc[valid, SCORES].to_numpy()).all():
        raise ValueError("Missing/nonfinite scores for scorable attempts")
    if scored.loc[~valid, SCORES].notna().any().any():
        raise ValueError("Invalid attempts must not carry predictor scores")
    rows = []
    for key, group in scored.groupby(KEYS, sort=False):
        row = dict(zip(KEYS, key))
        scorable = validated.loc[group.index, "unscored_reason"].eq("")
        row.update(
            generated=len(group),
            scorable=int(scorable.sum()),
            unscored=int((~scorable).sum()),
            unique_scorable=group.loc[scorable, "peptide"].nunique(),
        )
        reasons = validated.loc[group.index, "unscored_reason"]
        canonical = validated.loc[group.index, "canonical_9mer"]
        row.update(
            excluded_short=int(reasons.eq("too_short").sum()),
            unscored_X=int(group.peptide.map(lambda p: "X" in p and len(p) >= 8).sum()),
            canonical_9mers=int(canonical.sum()),
            mean_presentation_score=float(group.presentation_score.mean())
            if scorable.any()
            else None,
            mean_presentation_percentile=float(group.presentation_percentile.mean())
            if scorable.any()
            else None,
        )
        for cutoff in CUTOFFS:
            hits = group.presentation_percentile.lt(cutoff)
            label = f"ps_rank_lt_{cutoff:g}"
            row[label + "_count"] = int(hits.sum())
            row[label + "_rate"] = float(hits.sum() / len(group))
            row[label + "_unique_count"] = int(group.loc[hits, "peptide"].nunique())
            row[label + "_conditional_rate"] = (
                float(hits[scorable].mean()) if scorable.any() else None
            )
            row[label + "_canonical_rate"] = (
                float(hits[canonical].mean()) if canonical.any() else None
            )
        for cutoff in (100, 500):
            row[f"affinity_lt_{cutoff}_rate"] = float(
                group.affinity.lt(cutoff).sum() / len(group)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_existing(
    settings: dict, run_dir: Path, *, device: str, threads: int
) -> dict:
    """Load user-installed weights; evaluate existing CSV only, without generation.

    Parameters
    ----------
    settings : dict
        Explicit input, HLA list, installed-model path and panel sizes.
    run_dir : Path
        Shared-runtime output directory.
    device : str
        Explicit ``cpu`` or ``cuda``; CUDA requests must be available.
    threads : int
        Evaluator thread limit.

    Returns
    -------
    dict
        Input identities and execution summary; result CSVs are written to run_dir.
    """
    if not settings["input"] or not settings["hla_file"] or not settings["models_dir"]:
        raise ValueError(
            "Evaluation requires input, hla_file, and the installed models_dir"
        )
    source, hla_file = Path(settings["input"]), Path(settings["hla_file"])
    hlas = load_evaluation_hlas(hla_file)
    if len(hlas) != settings["expected_hlas"]:
        raise ValueError("Unexpected evaluation HLA panel size")
    frame = pd.read_csv(source, keep_default_na=False)
    validate_attempts(frame, hlas, settings["attempts_per_hla"])
    from mhcflurry import Class1PresentationPredictor
    from mhcflurry.common import configure_pytorch

    if device not in ("cpu", "cuda"):
        raise ValueError("MHCflurry evaluation device must be cpu or cuda")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA evaluation is unavailable")
    configure_pytorch(backend="gpu" if device == "cuda" else "cpu", num_threads=threads)
    predictor = Class1PresentationPredictor.load(
        models_dir=str(Path(settings["models_dir"]).resolve())
    )
    scored = score_attempts(frame, predictor, hlas, settings["attempts_per_hla"])
    metrics = aggregate_attempts(scored, hlas, settings["attempts_per_hla"])
    totals, summary = summarize_panels(metrics, hlas, settings["attempts_per_hla"])
    scored.to_csv(run_dir / "predictions.csv.gz", index=False)
    metrics.to_csv(run_dir / "per_hla.csv", index=False)
    totals.to_csv(run_dir / "per_seed.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    result = {
        "status": "complete",
        "stage": "evaluate",
        "input_sha256": sha256(source),
        "hla_file_sha256": sha256(hla_file),
        "attempts": len(frame),
        "groups": len(metrics),
        "package": importlib.metadata.version("mhcflurry"),
        "reference_weight_release": "2.3.0",
        "models_dir": str(Path(settings["models_dir"]).resolve()),
        "no_flanks": True,
        "metric": "presentation_percentile < 0.5; all attempts denominator",
        "device": device,
        "netmhcpan_used": False,
        "generation_run": False,
        "predictions_sha256": sha256(run_dir / "predictions.csv.gz"),
    }
    write_json(run_dir / "evaluation_summary.json", result)
    return result


def summarize_panels(metrics: pd.DataFrame, hlas: list[str], attempts: int = 1000):
    """Compute seed totals and mean +/- SEM across seeds, not across HLA rows.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Complete per-condition/seed/HLA counts.
    hlas : list[str]
        Expected HLA panel.
    attempts : int
        Required denominator per HLA.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Seed totals and per-prior mean/SEM. One-seed SEM is undefined.
    """
    required = set(KEYS + ["generated", "scorable", "ps_rank_lt_0.5_count"])
    if not required <= set(metrics) or metrics.empty or metrics.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate per-HLA evaluation metrics")
    if not metrics.generated.eq(attempts).all():
        raise ValueError(
            "Per-HLA attempt denominator differs from the evaluation contract"
        )
    for _, group in metrics.groupby(["prior", "seed"]):
        if len(group) != len(hlas) or set(group.hla) != set(hlas):
            raise ValueError("Incomplete per-seed HLA panel")
    hits = metrics["ps_rank_lt_0.5_count"]
    if not (
        (0 <= hits)
        & (hits <= metrics.scorable)
        & (metrics.scorable <= metrics.generated)
    ).all():
        raise ValueError("Inconsistent screening-hit/scorable/attempt counts")
    if not np.equal(hits, hits.astype(int)).all():
        raise ValueError("Screening-hit counts must be integers")
    totals = (
        metrics.groupby(["prior", "seed"], sort=False)[
            ["generated", "scorable", "ps_rank_lt_0.5_count"]
        ]
        .sum()
        .reset_index()
    )
    rows = []
    for prior, group in totals.groupby("prior", sort=False):
        values = group["ps_rank_lt_0.5_count"]
        rows.append(
            {
                "prior": prior,
                "seeds": len(values),
                "mean": float(values.mean()),
                "sem": float(values.sem(ddof=1)) if len(values) > 1 else None,
                "attempts_per_seed": len(hlas) * attempts,
            }
        )
    return totals, pd.DataFrame(rows)
