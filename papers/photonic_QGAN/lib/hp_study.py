from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, KFold


def _setup_arch_grid() -> list[dict[str, Any]]:
    """Return supported photonic setups for hp_study case generation."""
    return [
        {
            "setup": "setup_c",
            "noise_dim": 1,
            "arch": ["var", "var", "enc[2]", "var", "var"],
        },
    ]


def _setup_arch_from_label(setup_label: str) -> dict[str, Any] | None:
    """Resolve a setup label to a normalized setup dict used by each case."""
    key = str(setup_label).strip().lower()
    for item in _setup_arch_grid():
        if item["setup"] == key:
            return {
                "setup": item["setup"],
                "noise_dim": int(item["noise_dim"]),
                "arch": list(item["arch"]),
            }
    return None


def _coerce_str_list(value: Any) -> list[str]:
    """Parse config values that may be list, JSON string, or CSV-like string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _coerce_int_list(value: Any) -> list[int]:
    """Parse config values into a list of ints from flexible input formats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [int(item) for item in loaded]
    return [int(loaded)]


def _coerce_input_state(value: Any) -> list[int] | None:
    """Parse an input state as bits, accepting '01010', JSON lists, or CSV-like text."""
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    if "," not in text and "[" not in text and "]" not in text and all(ch in "01" for ch in text):
        return [int(ch) for ch in text]
    return _coerce_int_list(value)


def _arch_is_valid_for_modes(arch: list[str], mode_count: int) -> bool:
    """Ensure all encoder indices in the architecture fit within mode_count."""
    for token in arch:
        if token.startswith("enc[") and token.endswith("]"):
            idx_text = token[4:-1]
            try:
                idx = int(idx_text)
            except ValueError:
                return False
            if idx < 0 or idx >= mode_count:
                return False
    return True


def _resolve_csv_path(csv_path: str | Path, data_root: str | Path | None, repo_root: Path, project_dir: Path) -> Path:
    """Resolve the dataset CSV path against project and data-root fallbacks."""
    path = Path(csv_path)
    if path.is_absolute():
        return path

    candidates: list[Path] = [(repo_root / "data" / path).resolve()]
    if data_root:
        candidates.append((Path(data_root) / path).resolve())
    candidates.append((project_dir / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "CSV file not found; tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _prepare_dataset(
    csv_path: str | Path,
    label: int,
    batch_size: int,
    opt_iter_num: int,
    data_root: str | Path | None,
    repo_root: Path,
    project_dir: Path,
):
    """Build a replayable random-sampled dataloader sized to opt_iter_num steps."""
    import torch
    from torch.utils.data import RandomSampler
    from torchvision import transforms

    from papers.shared.photonic_QGAN.digits import DigitsDataset

    resolved_csv = _resolve_csv_path(csv_path, data_root, repo_root, project_dir)
    dataset = DigitsDataset(
        csv_file=str(resolved_csv),
        label=label,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * opt_iter_num)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )


def _extract_ssim_score(ssim_progress: list[tuple[float, float, float]], tail: int) -> float:
    """Aggregate SSIM from QGAN training, using a tail average for stability."""
    if not ssim_progress:
        return float("-inf")
    values = np.asarray([row[2] for row in ssim_progress], dtype=float)
    tail_values = values[-tail:] if tail > 0 else values
    return float(np.mean(tail_values))


class QGANSSIMEstimator(BaseEstimator):
    """Scikit-learn estimator wrapper around QGAN training and SSIM evaluation.

    `HalvingGridSearchCV` mutates hyperparameters on this estimator and repeatedly
    calls `fit` (per fold/per resource level). `score` returns the aggregated SSIM
    computed during the latest `fit` call.
    """

    def __init__(
        self,
        cases: list[dict[str, Any]],
        csv_path: str,
        data_root: str | None,
        repo_root: str,
        project_dir: str,
        image_size: int = 8,
        batch_size: int = 4,
        lossy: bool = False,
        use_clements: bool = False,
        sim: bool = False,
        remote_token: str | None = None,
        opt_iter_num: int = 300,
        lrD: float = 0.0002,
        lrG: float = 0.002,
        adam_beta1: float = 0.5,
        adam_beta2: float = 0.999,
        real_label: float = 0.9,
        fake_label: float = 0.0,
        gen_target: float = 0.9,
        d_steps: int = 1,
        g_steps: int = 1,
        ssim_tail: int = 100,
        base_seed: int = 0,
    ) -> None:
        self.cases = cases
        self.csv_path = csv_path
        self.data_root = data_root
        self.repo_root = repo_root
        self.project_dir = project_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.lossy = lossy
        self.use_clements = use_clements
        self.sim = sim
        self.remote_token = remote_token
        self.opt_iter_num = opt_iter_num
        self.lrD = lrD
        self.lrG = lrG
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.real_label = real_label
        self.fake_label = fake_label
        self.gen_target = gen_target
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.ssim_tail = ssim_tail
        self.base_seed = base_seed

    def fit(self, X, y=None):
        import perceval as pcvl
        import torch

        from lib.qgan import QGAN

        fit_start = time.perf_counter()
        # Derive a deterministic seed from the CV split so fold evaluations are repeatable.
        fold_seed = self.base_seed + int(np.asarray(X).sum())
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        logger.info(
            "[hp_study] Candidate start: iters={} lrD={} lrG={} betas=({}, {}) steps=(d={}, g={}) cases={}",
            self.opt_iter_num,
            self.lrD,
            self.lrG,
            self.adam_beta1,
            self.adam_beta2,
            self.d_steps,
            self.g_steps,
            len(self.cases),
        )

        case_scores: list[dict[str, Any]] = []
        # Evaluate each scenario (setup/digit/input_state) and average them into one score.
        for case_idx, case in enumerate(self.cases, start=1):
            case_start = time.perf_counter()
            logger.info(
                "[hp_study] Case {}/{} start: setup={} digit={} input_state={}",
                case_idx,
                len(self.cases),
                case["setup"],
                case["digit"],
                "".join(str(int(x)) for x in case["input_state"]),
            )
            dataloader = _prepare_dataset(
                csv_path=self.csv_path,
                label=int(case["digit"]),
                batch_size=int(self.batch_size),
                opt_iter_num=int(self.opt_iter_num),
                data_root=self.data_root,
                repo_root=Path(self.repo_root),
                project_dir=Path(self.project_dir),
            )
            qgan = QGAN(
                int(self.image_size),
                int(case["gen_count"]),
                case["arch"],
                pcvl.BasicState(case["input_state"]),
                int(case["noise_dim"]),
                int(self.batch_size),
                bool(case["pnr"]),
                bool(self.lossy),
                remote_token=self.remote_token,
                use_clements=bool(self.use_clements),
                sim=bool(self.sim),
            )
            _, _, _, _, ssim_progress = qgan.fit(
                dataloader,
                float(self.lrD),
                float(self.lrG),
                int(self.opt_iter_num),
                train_params={
                    "adam_beta1": float(self.adam_beta1),
                    "adam_beta2": float(self.adam_beta2),
                    "real_label": float(self.real_label),
                    "fake_label": float(self.fake_label),
                    "gen_target": float(self.gen_target),
                    "d_steps": int(self.d_steps),
                    "g_steps": int(self.g_steps),
                },
                silent=True,
                callback=None,
            )
            case_score = _extract_ssim_score(ssim_progress, tail=int(self.ssim_tail))
            case_payload = dict(case)
            case_payload["ssim_score"] = case_score
            case_scores.append(case_payload)
            logger.info(
                "[hp_study] Case {}/{} done: score={:.6f} elapsed={:.2f}s",
                case_idx,
                len(self.cases),
                case_score,
                time.perf_counter() - case_start,
            )

        self.case_scores_ = case_scores
        # Halving selects/promotes candidates using this scalar score.
        self.score_ = float(np.mean([item["ssim_score"] for item in case_scores]))
        logger.info(
            "[hp_study] Candidate done: score={:.6f} elapsed={:.2f}s",
            self.score_,
            time.perf_counter() - fit_start,
        )
        return self

    def score(self, X, y=None):
        """Return the score produced by the latest `fit`; sklearn calls this after fit."""
        return float(getattr(self, "score_", float("-inf")))


def _default_param_grid() -> dict[str, list[Any]]:
    """Default hp_study search space used when config does not override param_grid."""
    return {
        "lrD": [0.0002, 0.0005, 0.001],
        "lrG": [0.001, 0.002, 0.004],
        "adam_beta1": [0.5, 0.7],
        "adam_beta2": [0.99, 0.999],
        "d_steps": [1, 2],
        "g_steps": [1, 2, 3],
        "real_label": [0.9],
        "gen_target": [0.9],
    }


def _normalize_param_grid(raw_grid: dict[str, Any]) -> dict[str, list[Any]]:
    """Normalize param_grid values so every key maps to a list (sklearn contract)."""
    normalized: dict[str, list[Any]] = {}
    for key, value in raw_grid.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def _build_cases(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand config into concrete study cases (setup x input_state x digit)."""
    study_cfg = cfg.get("hp_study", {})
    ideal_cfg = cfg.get("ideal", {})

    setups = _coerce_str_list(study_cfg.get("setups"))
    if not setups:
        setups = [str(ideal_cfg.get("setup", "setup_c"))]

    digits = _coerce_int_list(study_cfg.get("digits"))
    if not digits:
        digits = [0, 3]

    input_values = study_cfg.get("input_states")
    if input_values is None:
        input_values = [ideal_cfg.get("input_state", "01010")]
    elif not isinstance(input_values, list):
        input_values = [input_values]
    input_states = []
    for value in input_values:
        parsed = _coerce_input_state(value)
        if not parsed:
            continue
        input_states.append(parsed)
    if not input_states:
        raise ValueError("hp_study.input_states resolved to an empty list.")

    cases: list[dict[str, Any]] = []
    for setup_label in setups:
        setup_cfg = _setup_arch_from_label(setup_label)
        if setup_cfg is None:
            raise ValueError(
                f"Unknown setup {setup_label}. Expected setup_c."
            )
        for input_state in input_states:
            if not _arch_is_valid_for_modes(setup_cfg["arch"], len(input_state)):
                continue
            for digit in digits:
                case = {
                    "setup": setup_cfg["setup"],
                    "digit": int(digit),
                    "input_state": list(input_state),
                    "noise_dim": int(setup_cfg["noise_dim"]),
                    "arch": list(setup_cfg["arch"]),
                    "gen_count": int(ideal_cfg.get("gen_count", 4 if len(input_state) == 5 else 2)),
                    "pnr": bool(ideal_cfg.get("pnr", False)),
                }
                cases.append(case)
    if not cases:
        raise ValueError("No valid hp study cases were produced from setups/input_states.")
    return cases


def _write_csv_results(path: Path, cv_results: dict[str, Any]) -> None:
    """Persist selected sklearn CV result fields in a compact, analysis-friendly CSV."""
    fields = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "param_lrD",
        "param_lrG",
        "param_adam_beta1",
        "param_adam_beta2",
        "param_d_steps",
        "param_g_steps",
        "param_real_label",
        "param_gen_target",
        "param_opt_iter_num",
    ]
    rows = len(cv_results.get("mean_test_score", []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(rows):
            row = {}
            for field in fields:
                values = cv_results.get(field)
                row[field] = values[i] if values is not None else ""
            writer.writerow(row)


def run_halving_grid_study(cfg: dict[str, Any], run_dir: Path, log) -> dict[str, Any]:
    """Run successive-halving grid search and write ranking artifacts under run_dir/hp_study."""
    study_cfg = cfg.get("hp_study", {})
    search_cfg = study_cfg.get("halving", {})
    training_ideal = cfg.get("training", {}).get("ideal", {})
    model_cfg = cfg.get("model", {})

    cases = _build_cases(cfg)
    param_grid = _normalize_param_grid(study_cfg.get("param_grid", _default_param_grid()))

    max_resources = int(search_cfg.get("max_resources", int(training_ideal.get("opt_iter_num", 1500))))
    min_resources = int(search_cfg.get("min_resources", max(100, max_resources // 8)))
    if min_resources > max_resources:
        raise ValueError("hp_study.halving.min_resources must be <= max_resources.")

    cv_splits = int(search_cfg.get("cv", 3))
    if cv_splits < 2:
        raise ValueError("hp_study.halving.cv must be >= 2.")

    factor = int(search_cfg.get("factor", 3))
    aggressive_elimination = bool(search_cfg.get("aggressive_elimination", True))
    random_state = int(search_cfg.get("random_state", cfg.get("seed", 0)))
    ssim_tail = int(study_cfg.get("ssim_tail", 100))

    csv_path = str(cfg.get("dataset", {}).get("csv_path", "photonic_QGAN/optdigits_csv.csv"))
    data_root = cfg.get("data_root")
    repo_root = str(Path(__file__).resolve().parents[3])
    project_dir = str(Path(__file__).resolve().parents[1])

    estimator = QGANSSIMEstimator(
        cases=cases,
        csv_path=csv_path,
        data_root=data_root,
        repo_root=repo_root,
        project_dir=project_dir,
        image_size=int(model_cfg.get("image_size", 8)),
        batch_size=int(model_cfg.get("batch_size", 4)),
        lossy=bool(model_cfg.get("lossy", False)),
        use_clements=bool(model_cfg.get("use_clements", False)),
        sim=bool(model_cfg.get("sim", False)),
        remote_token=model_cfg.get("remote_token"),
        opt_iter_num=min_resources,
        lrD=float(training_ideal.get("lrD", 0.0002)),
        lrG=float(training_ideal.get("lrG", 0.002)),
        adam_beta1=float(training_ideal.get("adam_beta1", 0.5)),
        adam_beta2=float(training_ideal.get("adam_beta2", 0.999)),
        real_label=float(training_ideal.get("real_label", 0.9)),
        fake_label=float(training_ideal.get("fake_label", 0.0)),
        gen_target=float(training_ideal.get("gen_target", 0.9)),
        d_steps=int(training_ideal.get("d_steps", 1)),
        g_steps=int(training_ideal.get("g_steps", 1)),
        ssim_tail=ssim_tail,
        base_seed=random_state,
    )

    # Explicit KFold lets us control reproducibility and avoid sklearn defaults changing.
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = HalvingGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        factor=factor,
        aggressive_elimination=aggressive_elimination,
        resource="opt_iter_num",
        min_resources=min_resources,
        max_resources=max_resources,
        cv=kfold,
        scoring=None,
        refit=True,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )

    # Placeholder feature matrix: estimator ignores actual X/y values, but sklearn CV requires them.
    X = np.arange(max(2 * cv_splits, 8), dtype=float).reshape(-1, 1)
    y = np.zeros(X.shape[0], dtype=int)

    log.info(
        "Starting halving grid search candidates={} cv={} resource=opt_iter_num [{}, {}] factor={} cases={}",
        np.prod([len(values) for values in param_grid.values()]),
        cv_splits,
        min_resources,
        max_resources,
        factor,
        len(cases),
    )
    log.info("Detailed hp_study progress logs will appear in the run log with [hp_study] prefix.")
    # Main optimization run; may take hours for large candidate sets/cases/resources.
    search.fit(X, y)

    study_out = run_dir / "hp_study"
    study_out.mkdir(parents=True, exist_ok=True)
    _write_csv_results(study_out / "cv_results.csv", search.cv_results_)

    top_k = int(study_cfg.get("top_k", 10))
    ranked_indices = np.argsort(search.cv_results_["rank_test_score"])[:top_k]
    top_candidates = []
    for idx in ranked_indices:
        top_candidates.append(
            {
                "rank": int(search.cv_results_["rank_test_score"][idx]),
                "score": float(search.cv_results_["mean_test_score"][idx]),
                "params": search.cv_results_["params"][idx],
            }
        )

    # Structured summary consumed by downstream analysis/scripts.
    payload = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_index": int(search.best_index_),
        "cases": cases,
        "top_candidates": top_candidates,
        "halving": {
            "min_resources": min_resources,
            "max_resources": max_resources,
            "factor": factor,
            "aggressive_elimination": aggressive_elimination,
            "cv": cv_splits,
            "random_state": random_state,
        },
    }
    (study_out / "best_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    training_patch = dict(training_ideal)
    training_patch.update(search.best_params_)
    (study_out / "best_training_ideal.json").write_text(
        json.dumps(training_patch, indent=2), encoding="utf-8"
    )

    log.info(
        "Halving study complete best_score={:.6f} best_params={}",
        search.best_score_,
        search.best_params_,
    )
    return payload
