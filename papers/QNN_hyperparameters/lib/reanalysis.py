"""E-A: re-analysis of the author's released per-configuration results.

Reads ``reports/results/*.csv`` of the author repository (commit 53bc59a),
builds one tidy table, and regenerates:

- E-A1: top-five configurations by weighted F1 per dataset and noise setting
  (paper Tables 2-5), accuracy ranges, and noisy-vs-noiseless paired
  accuracy differences (Sec. 5.1, Fig. 3);
- E-A2: the significance tests of the author notebooks
  (``notebooks/analyze_plain_results_*.ipynb``): Friedman plus one-sided
  Wilcoxon for paired factors (initialization, optimizer), Kruskal-Wallis for
  unpaired ones (ansatz, entanglement), Mann-Whitney U for preprocessing.
  Groups are paired by sorting on every other configuration column, as the
  notebooks do;
- E-A3: the initialization equivalence classes. If the initialization is never
  applied (author ``opt.py``), then ``zero``, ``uniform`` and ``normal`` are
  the same procedure (raw inputs), and so are ``beta``, ``beta_mu`` and
  ``normal_beta_dist`` (MinMax inputs). Paired differences within a class
  should be centred on zero; differences across classes should not.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon

logger = logging.getLogger(__name__)

FILE_PATTERN = re.compile(
    r"(?P<dataset>covtype|kdd|glass_identification|rice)_(?P<noise>ibm_perth|_)_"
    r"(?P<optimizer>cobyla|spsa|neldermead)_(?P<preprocessing>pca|lda)_(?P<init>.+)_results\.csv"
)
FUNCTION_NAMES = {
    "get_efficientsu2": "EfficientSU2",
    "get_paulitwodesign": "PauliTwoDesign",
    "get_realamplitudes": "RealAmplitudes",
    "get_twolocal": "TwoLocal",
    "get_zfeaturemap": "ZFeatureMap",
    "get_zzfeaturemap": "ZZFeatureMap",
}
CONFIG_COLUMNS = [
    "ansatz",
    "featuremap",
    "entanglement",
    "entanglement_featuremap",
    "preprocessing",
    "optimizer",
    "init",
]
RAW_INPUT_INITS = ("uniform", "normal", "zero")
MINMAX_INPUT_INITS = ("beta", "beta_mu", "normal_beta_dist")


def _function_name(text: str) -> str:
    match = re.search(r"<function (\w+) at", str(text))
    if not match:
        raise ValueError(f"cannot parse function reference {text!r}")
    return FUNCTION_NAMES[match.group(1)]


def load_released_results(results_dir: Path) -> pd.DataFrame:
    """Load every released CSV into one tidy frame.

    Parameters
    ----------
    results_dir : pathlib.Path
        ``<author_repo>/reports/results``.

    Returns
    -------
    pandas.DataFrame
        One row per released run with columns ``dataset``, ``noisy``,
        ``optimizer``, ``preprocessing``, ``init``, ``index`` (position in the
        author enumeration), ``ansatz``, ``featuremap``, ``entanglement``,
        ``entanglement_featuremap``, ``accuracy``, ``f1``, ``time``,
        ``iterations`` (objective evaluations) and ``source_file``.
    """

    frames = []
    for path in sorted(results_dir.glob("*_results.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected result file name {path.name}")
        frame = pd.read_csv(path)
        frame["dataset"] = match["dataset"]
        frame["noisy"] = match["noise"] == "ibm_perth"
        frame["optimizer"] = match["optimizer"]
        frame["preprocessing"] = match["preprocessing"]
        frame["init"] = match["init"]
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no released result files in {results_dir}")
    results = pd.concat(frames, ignore_index=True)
    # Failed author runs have empty metrics (``train_vqc`` returns None on
    # error); they become NaN and drop out of every paired comparison.
    for column in ("accuracy", "f1", "time", "iterations"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results["ansatz"] = results["ansatz"].map(_function_name)
    results["featuremap"] = results["featuremap"].map(_function_name)
    return results


def top_configurations(results: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Top ``n`` configurations by weighted F1 per dataset and noise setting.

    The paper's Tables 2-5 use the uniform, normal and beta arms; the post-hoc
    arms (beta_mu, normal_beta_dist, zero) and the older ``True`` files are
    excluded here, as they are not part of the main grid.
    """

    main = results[results["init"].isin(["uniform", "normal", "beta"])]
    ranked = (
        main.sort_values("f1", ascending=False)
        .groupby(["dataset", "noisy"], sort=True)
        .head(n)
    )
    return ranked.sort_values(
        ["dataset", "noisy", "f1"], ascending=[True, True, False]
    )[
        [
            "dataset",
            "noisy",
            "accuracy",
            "f1",
            "time",
            "ansatz",
            "entanglement",
            "optimizer",
            "featuremap",
            "entanglement_featuremap",
            "preprocessing",
            "init",
        ]
    ]


def accuracy_ranges(results: pd.DataFrame) -> pd.DataFrame:
    """Minimum and maximum accuracy per dataset and noise setting (main grid)."""

    main = results[results["init"].isin(["uniform", "normal", "beta"])]
    return (
        main.groupby(["dataset", "noisy"])["accuracy"]
        .agg(["min", "max", "count"])
        .reset_index()
    )


def _paired(frame: pd.DataFrame, factor: str, level: str, keys: list[str]) -> pd.Series:
    subset = frame[frame[factor] == level]
    return subset.set_index(keys)["accuracy"].dropna()


def noise_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Paired |noiseless - noisy| accuracy per dataset (paper Sec. 5.1)."""

    keys = ["dataset", "init", *[c for c in CONFIG_COLUMNS if c != "init"]]
    main = results[results["init"].isin(["uniform", "normal", "beta"])]
    clean = main[~main["noisy"]].set_index(keys)["accuracy"]
    noisy = main[main["noisy"]].set_index(keys)["accuracy"]
    joined = pd.concat({"noiseless": clean, "noisy": noisy}, axis=1).dropna()
    joined["abs_diff"] = (joined["noiseless"] - joined["noisy"]).abs()
    return (
        joined.reset_index()
        .groupby("dataset")["abs_diff"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )


def _wilcoxon(a: pd.Series, b: pd.Series, alternative: str = "two-sided") -> dict:
    joined = pd.concat({"a": a, "b": b}, axis=1).dropna()
    difference = joined["a"] - joined["b"]
    result = {
        "n_pairs": int(len(joined)),
        "mean_diff": float(difference.mean()),
        "median_diff": float(difference.median()),
    }
    if len(joined) == 0 or np.allclose(difference, 0):
        result["p_value"] = 1.0
        return result
    result["p_value"] = float(
        wilcoxon(joined["a"], joined["b"], alternative=alternative).pvalue
    )
    return result


def paired_factor_tests(
    results: pd.DataFrame, factor: str, levels: list[str], better: str | None
) -> list[dict]:
    """Friedman over ``levels`` plus pairwise Wilcoxon, per dataset and noise.

    Pairs are formed on every configuration column except ``factor`` (author
    notebooks: ``set_index(index).sort_index()``). With ``better`` set, the
    comparisons ``better`` vs each other level are one-sided ``greater``, as
    in the notebooks.
    """

    rows = []
    keys = [c for c in CONFIG_COLUMNS if c != factor]
    for (dataset, noisy), frame in results.groupby(["dataset", "noisy"]):
        series = {level: _paired(frame, factor, level, keys) for level in levels}
        available = [level for level in levels if len(series[level])]
        if len(available) < 2:
            continue
        aligned = pd.concat(
            {level: series[level] for level in available}, axis=1
        ).dropna()
        row = {
            "dataset": dataset,
            "noisy": bool(noisy),
            "factor": factor,
            "levels": available,
            "n_aligned": int(len(aligned)),
        }
        if len(available) >= 3 and len(aligned):
            row["friedman_p"] = float(
                friedmanchisquare(*[aligned[level] for level in available]).pvalue
            )
        row["pairwise"] = {}
        for first, second in combinations(available, 2):
            if better in (first, second):
                lead, other = (first, second) if first == better else (second, first)
                row["pairwise"][f"{lead}>{other}"] = _wilcoxon(
                    series[lead], series[other], alternative="greater"
                )
            else:
                row["pairwise"][f"{first}~{second}"] = _wilcoxon(
                    series[first], series[second]
                )
        rows.append(row)
    return rows


def unpaired_factor_tests(results: pd.DataFrame, factor: str) -> list[dict]:
    """Kruskal-Wallis (more than two levels) or Mann-Whitney U per dataset and noise."""

    rows = []
    main = results[results["init"].isin(["uniform", "normal", "beta"])]
    for (dataset, noisy), frame in main.groupby(["dataset", "noisy"]):
        groups = {
            str(level): group["accuracy"].dropna().to_numpy()
            for level, group in frame.groupby(factor, dropna=False)
        }
        groups = {level: values for level, values in groups.items() if len(values)}
        if len(groups) < 2:
            continue
        values = list(groups.values())
        test = kruskal(*values) if len(values) > 2 else mannwhitneyu(*values)
        rows.append(
            {
                "dataset": dataset,
                "noisy": bool(noisy),
                "factor": factor,
                "test": "kruskal" if len(values) > 2 else "mannwhitneyu",
                "p_value": float(test.pvalue),
                "means": {
                    level: float(np.mean(group)) for level, group in groups.items()
                },
            }
        )
    return rows


def init_equivalence(results: pd.DataFrame, margin: float = 0.02) -> list[dict]:
    """E-A3: paired differences within and across the two input-scaling classes.

    For each pair of initialization arms and each dataset (noiseless), report
    the paired mean difference, its bootstrap 95 % CI, a two-sided Wilcoxon p,
    and a TOST equivalence verdict at ``+-margin`` (both one-sided Wilcoxon
    tests on the shifted differences reject at 0.05).
    """

    rows = []
    keys = [c for c in CONFIG_COLUMNS if c != "init"]
    noiseless = results[~results["noisy"]]
    rng = np.random.default_rng(0)
    arms = [*RAW_INPUT_INITS, *MINMAX_INPUT_INITS]
    for dataset, frame in noiseless.groupby("dataset"):
        for first, second in combinations(arms, 2):
            a = _paired(frame, "init", first, keys)
            b = _paired(frame, "init", second, keys)
            joined = pd.concat({"a": a, "b": b}, axis=1).dropna()
            if len(joined) < 10:
                continue
            difference = (joined["a"] - joined["b"]).to_numpy()
            boot = rng.choice(
                difference, size=(2000, len(difference)), replace=True
            ).mean(axis=1)
            nonzero = not np.allclose(difference, 0)
            lower_p = (
                float(wilcoxon(difference + margin, alternative="greater").pvalue)
                if nonzero
                else 0.0
            )
            upper_p = (
                float(wilcoxon(difference - margin, alternative="less").pvalue)
                if nonzero
                else 0.0
            )
            same_class = (first in RAW_INPUT_INITS) == (second in RAW_INPUT_INITS)
            rows.append(
                {
                    "dataset": dataset,
                    "pair": f"{first} - {second}",
                    "same_input_scaling": same_class,
                    "n_pairs": int(len(difference)),
                    "mean_diff": float(difference.mean()),
                    "ci95": [
                        float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5)),
                    ],
                    "wilcoxon_p": float(wilcoxon(difference).pvalue)
                    if nonzero
                    else 1.0,
                    "tost_equivalent": bool(lower_p < 0.05 and upper_p < 0.05),
                    "margin": margin,
                }
            )
    return rows


def resolve_true_files(results: pd.DataFrame) -> list[dict]:
    """Match the older ``*_True_results.csv`` files to a named arm.

    Those files (Glass and KDD, noiseless) hold 5 repetitions of each of the
    84 configurations. Their per-configuration mean is compared with every
    named initialization arm; the closest arm identifies what they ran.
    """

    rows = []
    keys = [c for c in CONFIG_COLUMNS if c != "init"]
    noiseless = results[~results["noisy"]]
    for dataset, frame in noiseless.groupby("dataset"):
        repeated = frame[frame["init"] == "True"]
        if not len(repeated):
            continue
        true_mean = repeated.groupby(keys, dropna=False)["accuracy"].mean().dropna()
        for arm in ["uniform", "normal", "beta", "beta_mu", "normal_beta_dist", "zero"]:
            other = _paired(frame, "init", arm, keys)
            joined = pd.concat({"a": true_mean, "b": other}, axis=1).dropna()
            if len(joined):
                rows.append(
                    {
                        "dataset": dataset,
                        "compared_with": arm,
                        "n_pairs": int(len(joined)),
                        "mean_abs_diff": float(
                            (joined["a"] - joined["b"]).abs().mean()
                        ),
                        "mean_diff": float((joined["a"] - joined["b"]).mean()),
                    }
                )
    return rows


def repetition_spread(results: pd.DataFrame) -> list[dict]:
    """Run-to-run spread from the 5-repetition ``True`` files.

    For each dataset/optimizer/preprocessing, report the mean over
    configurations of the per-configuration accuracy standard deviation and of
    the expected |difference| between two independent repetitions. This is
    the noise floor against which single-run paired differences are judged.
    """

    rows = []
    keys = [c for c in CONFIG_COLUMNS if c != "init"]
    repeated = results[(results["init"] == "True") & (~results["noisy"])]
    for (dataset, optimizer, preprocessing), frame in repeated.groupby(
        ["dataset", "optimizer", "preprocessing"]
    ):
        grouped = frame.groupby(keys, dropna=False)["accuracy"]
        spreads = grouped.std(ddof=1).dropna()
        pair_gaps = grouped.apply(
            lambda values: (
                float(
                    np.mean([abs(a - b) for a, b in combinations(values.dropna(), 2)])
                )
                if values.notna().sum() > 1
                else np.nan
            )
        ).dropna()
        rows.append(
            {
                "dataset": dataset,
                "optimizer": optimizer,
                "preprocessing": preprocessing,
                "configs": int(len(spreads)),
                "repetitions_per_config": int(grouped.count().median()),
                "mean_sd": float(spreads.mean()),
                "mean_abs_gap_between_repetitions": float(pair_gaps.mean()),
            }
        )
    return rows
