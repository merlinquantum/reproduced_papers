from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .config import deep_update, load_config
from .utils import import_callable

TYPE_FACTORIES: dict[str, Callable[[Any], Any]] = {}


def _register_default_types() -> None:
    def _csv_int_list(value: Any) -> list[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        text = str(value).strip()
        if not text:
            return []
        return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]

    def _csv_int_matrix(value: Any) -> list[list[int]]:
        if isinstance(value, list):
            return [[int(cell) for cell in row] for row in value]
        text = str(value).strip()
        if not text:
            return []
        rows = [row.strip() for row in text.split(";") if row.strip()]
        return [
            [int(cell.strip()) for cell in row.split(",") if cell.strip()]
            for row in rows
        ]

    def _int_or_none(raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        text = str(raw).strip().lower()
        if text in {"none", "null", ""}:
            return None
        return int(raw)

    def _path_type(raw: Any) -> Path:
        return Path(raw)

    def _bool_type(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot interpret boolean value from: {raw}")

    TYPE_FACTORIES.update(
        {
            "int": int,
            "float": float,
            "str": str,
            "path": _path_type,
            "csv_int_list": _csv_int_list,
            "csv_int_matrix": _csv_int_matrix,
            "int_or_none": _int_or_none,
            "bool": _bool_type,
        }
    )


_register_default_types()


def register_cli_type(name: str, factory: Callable[[Any], Any]) -> None:
    TYPE_FACTORIES[name] = factory


def _derive_dest(flags: list[str]) -> str:
    for flag in flags:
        if flag.startswith("--"):
            return flag.lstrip("-").replace("-", "_")
    clean = flags[0].lstrip("-")
    return clean.replace("-", "_")


def _set_by_path(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = cfg
    segments = dotted_path.split(".")
    for key in segments[:-1]:
        if not isinstance(cursor.get(key), dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[segments[-1]] = value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(project_dir: Path, invocation_dir: Path, raw: str | Path) -> Path:
    path_value = Path(raw)
    if path_value.is_absolute():
        return path_value

    invocation_candidate = invocation_dir / path_value
    if invocation_candidate.exists():
        return invocation_candidate

    return project_dir / path_value


def build_cli_parser(
    cli_schema: dict[str, Any],
) -> tuple[argparse.ArgumentParser, list[dict[str, Any]]]:
    parser = argparse.ArgumentParser(
        description=cli_schema.get("description", "Paper reproduction runner"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed: list[dict[str, Any]] = []
    for entry in cli_schema.get("arguments", []):
        arg_def = dict(entry)
        flags = arg_def.get("flags")
        if not flags:
            raise ValueError("Each CLI entry requires at least one flag")
        dest = arg_def.get("dest") or _derive_dest(flags)
        arg_def["_dest"] = dest
        kwargs: dict[str, Any] = {"dest": dest}
        for key in ("help", "default", "choices", "metavar", "required", "nargs"):
            if key in arg_def:
                kwargs[key] = arg_def[key]
        action = arg_def.get("action")
        if action:
            kwargs["action"] = action
        else:
            type_name = arg_def.get("type")
            if type_name:
                try:
                    kwargs["type"] = TYPE_FACTORIES[type_name]
                except KeyError as exc:  # pragma: no cover
                    raise ValueError(f"Unsupported CLI type '{type_name}'") from exc
        parser.add_argument(*flags, **kwargs)
        processed.append(arg_def)
    return parser, processed


def apply_cli_overrides(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    arg_defs: list[dict[str, Any]],
    project_dir: Path,
    invocation_dir: Path,
) -> dict[str, Any]:
    for phase in ("config_file", "other"):
        for arg_def in arg_defs:
            apply_spec = arg_def.get("apply") or {}
            kind = apply_spec.get("kind", "config_path")
            if (phase == "config_file" and kind != "config_file") or (
                phase == "other" and kind == "config_file"
            ):
                continue
            dest = arg_def["_dest"]
            value = getattr(args, dest)
            if value is None:
                continue
            action = arg_def.get("action")
            if action == "store_true" and not value:
                continue
            if action == "store_false" and value:
                continue
            if kind == "config_file":
                cfg_path = _resolve_path(project_dir, invocation_dir, value)
                file_cfg = load_config(cfg_path)
                cfg = deep_update(cfg, file_cfg)
                continue
            if kind == "custom_callable":
                fn = import_callable(apply_spec["callable"])
                cfg = fn(cfg, value, arg_def)
                continue
            dotted_path = apply_spec.get("path")
            if not dotted_path:
                raise ValueError(
                    f"Missing 'path' for config override on CLI flag {arg_def.get('flags')}"
                )
            stored_value = apply_spec.get("value", value)
            _set_by_path(cfg, dotted_path, _json_safe(stored_value))
    return cfg


__all__ = [
    "TYPE_FACTORIES",
    "register_cli_type",
    "build_cli_parser",
    "apply_cli_overrides",
]
