"""Custom CLI overrides for experiment-scoped parameters."""

from __future__ import annotations

from typing import Any

from lib.runner import DEFAULT_EXPERIMENT, EXPERIMENT_SPECS

_PARAM_TARGETS: dict[str, set[str]] = {
    "input_state": {"accuracy_vs_kernel"},
    "input_states": {"accuracy_vs_input_state"},
    "superset_size": {
        "accuracy_vs_kernel",
        "accuracy_vs_input_state",
        "accuracy_vs_width",
    },
    "reg": {
        "accuracy_vs_kernel",
        "accuracy_vs_input_state",
        "accuracy_vs_width",
        "accuracy_vs_geometric_difference",
    },
    "reps": {"accuracy_vs_kernel", "accuracy_vs_input_state", "accuracy_vs_width"},
    "test_size": {
        "accuracy_vs_kernel",
        "accuracy_vs_input_state",
        "accuracy_vs_width",
        "accuracy_vs_geometric_difference",
    },
    "indistinguishability": {"accuracy_vs_kernel", "accuracy_vs_input_state"},
    "no_bunching": {"accuracy_vs_kernel", "accuracy_vs_input_state"},
    "force_psd": {"accuracy_vs_kernel", "accuracy_vs_input_state"},
    "shots": {"accuracy_vs_kernel", "accuracy_vs_input_state"},
    "seed": {
        "accuracy_vs_kernel",
        "accuracy_vs_input_state",
        "accuracy_vs_width",
        "accuracy_vs_geometric_difference",
    },
    "data_sizes": {
        "accuracy_vs_kernel",
        "accuracy_vs_input_state",
        "accuracy_vs_width",
    },
    "widths": {"accuracy_vs_width"},
    "num_points": {"accuracy_vs_geometric_difference"},
    "n": {"accuracy_vs_geometric_difference"},
    "data_size": {"accuracy_vs_geometric_difference"},
    "m_min": {"accuracy_vs_geometric_difference"},
    "m_max": {"accuracy_vs_geometric_difference"},
    "spline_smoothing": {"accuracy_vs_geometric_difference"},
}


def _canonicalize_experiment(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _normalize_experiments(value: Any) -> list[str]:
    if value is None:
        return [DEFAULT_EXPERIMENT]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [DEFAULT_EXPERIMENT]
        if text.lower() == "all":
            return list(EXPERIMENT_SPECS)
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [_canonicalize_experiment(item) for item in items]

    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if len(items) == 1 and items[0].lower() == "all":
            return list(EXPERIMENT_SPECS)
        return [_canonicalize_experiment(item) for item in items]

    return [_canonicalize_experiment(str(value))]


def apply_experiment_param(
    cfg: dict[str, Any], value: Any, arg_def: dict[str, Any]
) -> dict[str, Any]:
    """Apply a CLI value to the selected experiment(s)."""
    apply_spec = arg_def.get("apply") or {}
    param = apply_spec.get("param")
    if not param:
        raise ValueError("Missing 'param' for experiment-scoped CLI override")

    experiments = _normalize_experiments(cfg.get("experiment"))
    supported = _PARAM_TARGETS.get(param)
    if supported is None:
        raise ValueError(f"Unsupported experiment parameter: {param}")

    invalid = [name for name in experiments if name not in supported]
    if invalid:
        supported_list = ", ".join(sorted(supported))
        invalid_list = ", ".join(sorted(invalid))
        raise ValueError(
            f"Parameter '{param}' is not valid for experiment(s): {invalid_list}. "
            f"Supported experiments: {supported_list}"
        )

    experiments_cfg = cfg.setdefault("experiments", {})
    for name in experiments:
        exp_cfg = experiments_cfg.setdefault(name, {})
        exp_cfg[param] = value

    return cfg


__all__ = ["apply_experiment_param"]
