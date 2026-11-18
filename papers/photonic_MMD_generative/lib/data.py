# Dataset recipes follow merlinquantum/generative_classical_training (src/data/*.py),
# commit 17a12c4, Copyright (c) 2026 merlinquantum, MIT License.
"""Datasets of fixed-Hamming-weight bitstrings (paper §5) and the data loader.

Raw sources live under ``<data_root>/photonic_MMD_generative/`` (see LOG.md
"Data Acquisition Log"). Each builder writes a prepared copy to
``<data_root>/photonic_MMD_generative/prepared/<dataset_id>/`` with
``train.npy``, ``test.npy`` and ``manifest.json`` (recipe, row counts,
SHA-256 of the arrays), and reuses it when the recipe matches.

Datasets (``cfg["dataset"]["name"]``):
- ``boson_sampling``: Haar U (numpy seed) + Perceval Clifford2017 samples
  (paper §5.1). Occupations may exceed 1 (collisions); the manifest records
  the collision fraction (D5).
- ``sushi_top10``: PrefLib 00014-00000002.soi, top-10 of 100 sushi, seed 42,
  3000 / 2000 (§5.2).
- ``pairwise_sushi``: Kamishima sushi3a order file, 10 random item pairs
  (seed 42), one-hot pair blocks padded to m modes, 3000 / 2000 (§5.4).
- ``mushroom_blocks``: UCI Mushroom, one-hot of the first 10 categorical
  columns padded to m modes, seed 0, 5000 / 1000 (§5.4).
- ``movies``: Netflix ratings, top-m most rated movies, top-n per user from
  2005 on, seed 42, 10 000 / 10 000 (§5.2). Rebuilt with the authors'
  script logic; unverified against the authors' files (LOG.md).
- ``bioinformatics``: LINCS 2020 level-5 trt_xpr landmark genes, seed-42
  subset of 10 000 signatures, MAD gene selection, top-n "up" genes, 70 / 30
  (§5.3). Rebuilt with the authors' script logic; unverified.
- ``random_fixed_weight``: synthetic uniform fixed-weight bitstrings, for
  smoke runs only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PAPER_DATA_DIRNAME = "photonic_MMD_generative"
DATASETS = (
    "boson_sampling",
    "sushi_top10",
    "pairwise_sushi",
    "mushroom_blocks",
    "movies",
    "bioinformatics",
    "random_fixed_weight",
)
INPUT_STATE_TYPES = (
    "beginning",
    "end",
    "beginning_alternating",
    "middle_compact",
    "middle_alternating",
)


def input_modes(m: int, n: int, input_state: str) -> list[int]:
    """Occupied input modes for the original code's named input states."""
    if input_state == "beginning":
        return list(range(n))
    if input_state == "end":
        return list(range(m - n, m))
    if input_state == "beginning_alternating":
        return list(range(0, 2 * n - 1, 2))
    if input_state == "middle_compact":
        start = (m - n) // 2
        return list(range(start, start + n))
    if input_state == "middle_alternating":
        start = (m - (2 * n - 1)) // 2
        return list(range(start, start + 2 * n - 1, 2))
    raise ValueError(
        f"Unknown input state {input_state!r}; expected one of {INPUT_STATE_TYPES}."
    )


def paper_data_dir(cfg: dict) -> Path:
    """Return ``<data_root>/photonic_MMD_generative`` (data_root set by the runtime)."""
    return Path(cfg["data_root"]) / PAPER_DATA_DIRNAME


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _shuffle_split(X: np.ndarray, seed: int, n_train: int, n_test: int):
    X = np.array(X, copy=True)  # pandas 3 may hand back read-only views
    rng = np.random.default_rng(seed)
    rng.shuffle(X)
    return X[:n_train], X[n_train : n_train + n_test]


def _build_boson_sampling(spec: dict, data_dir: Path):
    import perceval as pcvl
    from scipy.stats import unitary_group

    m, n = spec["m"], spec["n"]
    U = unitary_group.rvs(m, random_state=np.random.default_rng(spec["unitary_seed"]))
    backend = pcvl.Clifford2017Backend()
    backend.set_circuit(pcvl.components.Unitary(pcvl.Matrix(U)))
    occupied = set(input_modes(m, n, spec["input_state"]))
    backend.set_input_state(
        pcvl.BasicState([1 if mode in occupied else 0 for mode in range(m)])
    )
    pcvl.random_seed(spec["sample_seed"])
    samples = np.array(
        [list(s) for s in backend.samples(spec["n_train"] + spec["n_test"])],
        dtype=np.int64,
    )
    if not (samples.sum(axis=1) == n).all():
        raise RuntimeError(
            "Clifford2017 returned samples with the wrong photon number."
        )
    train, test = _shuffle_split(
        samples, spec["sample_seed"], spec["n_train"], spec["n_test"]
    )
    collision_fraction = float((samples.max(axis=1) > 1).mean())
    return train, test, {"unitary": U, "collision_fraction": collision_fraction}


def _build_sushi_top10(spec: dict, data_dir: Path):
    path = data_dir / "preflib_sushi" / "00014-00000002.soi"
    rankings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # The authors' parser does not skip PrefLib '#' header lines; ours must.
        if not line.strip() or line.startswith("#"):
            continue
        rankings.append([int(item) for item in line.split(":", 1)[1].split(",")])
    rankings = np.asarray(rankings)
    X = np.zeros((len(rankings), spec["m"]), dtype=np.int64)
    X[np.arange(len(rankings))[:, None], rankings - 1] = 1
    train, test = _shuffle_split(X, spec["seed"], spec["n_train"], spec["n_test"])
    return train, test, {}


def _build_pairwise_sushi(spec: dict, data_dir: Path):
    path = (
        data_dir
        / "kamishima_sushi3"
        / "extracted"
        / "sushi3-2016"
        / "sushi3a.5000.10.order"
    )
    rankings = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if parts:
            rankings.append([int(v) for v in parts[2 : 2 + int(parts[1])]])
    rankings = np.asarray(rankings)
    all_pairs = list(combinations(range(spec["n_items"]), 2))
    chosen = np.random.default_rng(spec["seed"]).choice(
        len(all_pairs), size=spec["n_pairs"], replace=False
    )
    pairs = [all_pairs[i] for i in chosen]
    X = np.zeros((len(rankings), spec["m"]), dtype=np.int64)
    for row, ranking in enumerate(rankings):
        position = {item: pos for pos, item in enumerate(ranking)}
        for block, (a, b) in enumerate(pairs):
            X[row, 2 * block + (0 if position[a] < position[b] else 1)] = 1
    train, test = _shuffle_split(X, spec["seed"], spec["n_train"], spec["n_test"])
    return train, test, {"pairs": [list(p) for p in pairs]}


def _build_mushroom_blocks(spec: dict, data_dir: Path):
    import pandas as pd

    features = pd.read_csv(data_dir / "uci_mushroom" / "raw" / "data.csv").drop(
        columns=["poisonous"]
    )
    features = features.astype(str).replace("?", "missing")
    columns = list(features.columns[: spec["n_columns"]])
    blocks = [pd.get_dummies(features[c], prefix=c, dtype=int) for c in columns]
    block_sizes = [b.shape[1] for b in blocks]
    onehot = pd.concat(blocks, axis=1).to_numpy(dtype=np.int64)
    if onehot.shape[1] > spec["m"]:
        raise ValueError(f"{onehot.shape[1]} active modes exceed m={spec['m']}.")
    X = np.pad(onehot, ((0, 0), (0, spec["m"] - onehot.shape[1])))
    rng = np.random.default_rng(spec["seed"])
    subset = X[rng.permutation(len(X))[: spec["n_train"] + spec["n_test"]]]
    rng.shuffle(subset)
    train, test = (
        subset[: spec["n_train"]],
        subset[spec["n_train"] : spec["n_train"] + spec["n_test"]],
    )
    return train, test, {"block_sizes": block_sizes, "columns": columns}


def _build_movies(spec: dict, data_dir: Path):
    import pandas as pd

    path = (
        data_dir
        / "netflix_movie_ratings"
        / "raw"
        / "archive"
        / "Netflix_User_Ratings.csv"
    )
    m, n = spec["m"], spec["n"]
    dtypes = {"CustId": np.int32, "Rating": np.int8, "MovieId": np.int16, "Date": str}
    # Pass 1: rating counts over the full table (the script's active line uses df, not df_dated).
    counts = None
    for chunk in pd.read_csv(
        path, usecols=["MovieId"], dtype={"MovieId": np.int16}, chunksize=10_000_000
    ):
        c = chunk["MovieId"].value_counts()
        counts = c if counts is None else counts.add(c, fill_value=0)
    top_movies = counts.sort_values(ascending=False, kind="stable").head(m).index
    # Pass 2: rows from 2005 on for the top-m movies, original row order kept.
    kept = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=10_000_000):
        kept.append(
            chunk[
                (chunk["Date"] >= spec["min_date"]) & chunk["MovieId"].isin(top_movies)
            ]
        )
    df = pd.concat(kept, ignore_index=True)
    top_n = (
        df.sort_values(["CustId", "Rating", "Date"], ascending=[True, False, False])
        .groupby("CustId")
        .head(n)
    )
    distinct = top_n.groupby("CustId")["MovieId"].nunique()
    top_n = top_n[top_n["CustId"].isin(distinct[distinct == n].index)]
    onehot = (
        top_n[["CustId", "MovieId"]]
        .assign(value=1)
        .pivot_table(index="CustId", columns="MovieId", values="value", fill_value=0)
    )
    empty_columns = int(m - onehot.shape[1])
    onehot = onehot.reindex(columns=sorted(top_movies), fill_value=0)
    X = onehot.to_numpy(dtype=np.int64)
    if not (X.sum(axis=1) == n).all():
        raise RuntimeError("Movie bitstrings do not all have Hamming weight n.")
    train, test = _shuffle_split(X, spec["seed"], spec["n_train"], spec["n_test"])
    return (
        train,
        test,
        {"users_with_n_movies": int(len(X)), "movies_never_in_a_top_n": empty_columns},
    )


def _build_bioinformatics(spec: dict, data_dir: Path):
    import pandas as pd
    from cmapPy.pandasGEXpress.parse import parse

    raw = data_dir / "cmap_lincs2020"
    prepared_dir = data_dir / "prepared" / "bioinformatics_intermediate"
    subset_path = prepared_dir / "lincs2020_trt_xpr_lm978_subset10k.tsv"
    if not subset_path.exists():
        geneinfo = pd.read_csv(raw / "geneinfo_beta.txt", sep="\t")
        landmarks = geneinfo.loc[
            geneinfo["feature_space"] == "landmark", ["gene_symbol", "gene_id"]
        ]
        landmark_ids = [str(g) for g in landmarks.drop_duplicates()["gene_id"]]
        gct = parse(
            str(raw / "level5_beta_trt_xpr_n142901x12328.gctx"),
            convert_neg_666=True,
            rid=landmark_ids,
        )
        df_lm = gct.data_df
        subset_columns = np.random.default_rng(42).choice(
            df_lm.columns, size=10000, replace=False
        )
        prepared_dir.mkdir(parents=True, exist_ok=True)
        df_lm[subset_columns].to_csv(subset_path, sep="\t")
    df = pd.read_csv(subset_path, sep="\t", index_col=0).sort_index()
    samples = df.T
    train_set = samples.sample(frac=spec["frac_train"], random_state=42)
    test_set = samples.drop(train_set.index)
    median = train_set.median(axis=0)
    mad = train_set.sub(median, axis=1).abs().median(axis=0)
    universe = sorted(mad.sort_values(ascending=False).head(spec["m"]).index.tolist())
    n = spec["n"]

    def encode(frame):
        scores = frame[universe].to_numpy(dtype=float)
        top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
        X = np.zeros(scores.shape, dtype=np.int64)
        X[np.arange(len(X))[:, None], top] = 1
        return X

    n_train = int(spec["number_samples"] * spec["frac_train"])
    n_test = int(spec["number_samples"] * (1 - spec["frac_train"]))
    return (
        encode(train_set)[:n_train],
        encode(test_set)[:n_test],
        {"gene_universe": [str(g) for g in universe]},
    )


def _build_random_fixed_weight(spec: dict, data_dir: Path):
    rng = np.random.default_rng(spec["seed"])
    total = spec["n_train"] + spec["n_test"]
    X = np.zeros((total, spec["m"]), dtype=np.int64)
    X[
        np.arange(total)[:, None],
        np.argsort(rng.random((total, spec["m"])), axis=1)[:, : spec["n"]],
    ] = 1
    return X[: spec["n_train"]], X[spec["n_train"] :], {}


_BUILDERS = {
    "boson_sampling": _build_boson_sampling,
    "sushi_top10": _build_sushi_top10,
    "pairwise_sushi": _build_pairwise_sushi,
    "mushroom_blocks": _build_mushroom_blocks,
    "movies": _build_movies,
    "bioinformatics": _build_bioinformatics,
    "random_fixed_weight": _build_random_fixed_weight,
}


RECIPE_KEYS = {
    "boson_sampling": (
        "m",
        "n",
        "n_train",
        "n_test",
        "input_state",
        "unitary_seed",
        "sample_seed",
    ),
    "sushi_top10": ("m", "n", "n_train", "n_test", "seed"),
    "pairwise_sushi": ("m", "n", "n_items", "n_pairs", "n_train", "n_test", "seed"),
    "mushroom_blocks": ("m", "n", "n_columns", "n_train", "n_test", "seed"),
    "movies": ("m", "n", "n_train", "n_test", "seed", "min_date"),
    "bioinformatics": ("m", "n", "number_samples", "frac_train"),
    "random_fixed_weight": ("m", "n", "n_train", "n_test", "seed"),
}


def recipe(dataset_cfg: dict) -> dict:
    """Keep only the keys the named builder uses (config overlays deep-merge).

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    KeyError
        If a required recipe key is missing.
    """
    name = dataset_cfg["name"]
    if name not in RECIPE_KEYS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}.")
    missing = [key for key in RECIPE_KEYS[name] if key not in dataset_cfg]
    if missing:
        raise KeyError(f"Dataset {name!r} config is missing keys {missing}.")
    return {"name": name, **{key: dataset_cfg[key] for key in RECIPE_KEYS[name]}}


def dataset_id(spec: dict) -> str:
    """Stable identifier of a dataset recipe: name + short hash of the spec."""
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    return f"{spec['name']}_m{spec['m']}_n{spec['n']}_{digest}"


def load_dataset(cfg: dict) -> dict:
    """Build or load the prepared dataset described by ``cfg["dataset"]``.

    Returns
    -------
    dict
        ``train``, ``test`` (int64 arrays (N, m)), ``manifest`` (dict), and
        ``extras`` (builder-specific arrays, e.g. the BS unitary).

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    KeyError
        If a recipe key is missing.
    RuntimeError
        If the prepared arrays do not have the expected shape or weight.
    """
    spec = recipe(cfg["dataset"])
    data_dir = paper_data_dir(cfg)
    prepared = data_dir / "prepared" / dataset_id(spec)
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        train, test = np.load(prepared / "train.npy"), np.load(prepared / "test.npy")
        extras = {p.stem: np.load(p) for p in prepared.glob("extra_*.npy")}
        logger.info("Loaded prepared dataset %s", prepared)
    else:
        logger.info(
            "Building dataset %s (recipe %s)",
            spec["name"],
            json.dumps(spec, sort_keys=True),
        )
        train, test, info = _BUILDERS[spec["name"]](spec, data_dir)
        prepared.mkdir(parents=True, exist_ok=True)
        np.save(prepared / "train.npy", train)
        np.save(prepared / "test.npy", test)
        extras = {}
        for key, value in list(info.items()):
            if isinstance(value, np.ndarray):
                np.save(prepared / f"extra_{key}.npy", value)
                extras[f"extra_{key}"] = value
                info.pop(key)
        manifest = {
            "dataset_id": dataset_id(spec),
            "recipe": spec,
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "train_sha256": _sha256_array(train),
            "test_sha256": _sha256_array(test),
            "collision_free_train": bool(train.max() <= 1),
            **info,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Prepared dataset written to %s", prepared)
    m, n = spec["m"], spec["n"]
    for split, X in (("train", train), ("test", test)):
        if X.shape[1] != m or not (X.sum(axis=1) == n).all():
            raise RuntimeError(f"{split} split has shape {X.shape} or weights != {n}.")
    return {
        "train": train,
        "test": test,
        "manifest": manifest,
        "extras": extras,
        "path": str(prepared),
    }


def build_dataloader(cfg: dict, split: str):
    """Return a torch DataLoader over one split as float64 tensors.

    The MMD estimator uses the whole split at once, so the default batch is
    the full split (``cfg["dataset"]["batch_size"]`` overrides it).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    X = load_dataset(cfg)[split]
    tensor = torch.as_tensor(X, dtype=torch.float64, device=cfg["device"])
    batch_size = cfg["dataset"].get("batch_size") or len(X)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
