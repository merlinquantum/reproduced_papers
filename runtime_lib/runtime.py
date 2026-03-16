from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .cli import apply_cli_overrides, build_cli_parser
from .config import deep_update, load_config
from .data_paths import ENV_DATA_ROOT, resolve_data_root
from .dtypes import resolve_config_dtypes
from .logging_utils import configure_logging
from .seed import seed_everything
from .utils import ensure_no_placeholders, import_callable


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_RUNTIME_DIR = Path(__file__).resolve().parent
_GLOBAL_CLI_SCHEMA = _load_json(_RUNTIME_DIR / "global_cli.json")
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "seed": 1337,
    "dtype": None,
    "device": "cpu",
    "logging": {"level": "info"},
    "data_root": None,
}
_PROJECT_DEFAULTS_REL = Path("configs") / "defaults.json"
_PROJECT_CLI_SCHEMA_REL = Path("cli.json")
_DEFAULT_RUNNER_CALLABLE = "lib.runner.train_and_evaluate"
_DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
_DEFAULT_RUN_PATTERN = "run_{timestamp}"


def _log_run_banner(project_dir: Path, cfg: dict[str, Any]) -> None:
    logger = logging.getLogger("runtime")
    project_name = project_dir.name
    logger.info("Starting %s run", project_name)
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))


def _purge_project_modules() -> None:
    for name in list(sys.modules):
        if name == "lib" or name.startswith("lib."):
            sys.modules.pop(name, None)


def _ensure_project_on_path(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    _purge_project_modules()


def _prepare_runner_config(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = copy.deepcopy(cfg)
    resolve_config_dtypes(runtime_cfg)
    return runtime_cfg


def run_from_project(project_dir: Path, argv: list[str] | None = None) -> Path:
    project_dir = project_dir.resolve()
    invocation_dir = Path.cwd().resolve()
    configure_logging("info")
    if invocation_dir != project_dir:
        logging.info("Switching working directory to %s", project_dir)
        os.chdir(project_dir)
    _ensure_project_on_path(project_dir)

    cli_schema_path = project_dir / _PROJECT_CLI_SCHEMA_REL
    if not cli_schema_path.exists():
        raise FileNotFoundError(f"Missing CLI schema: {cli_schema_path}")
    cli_schema = _load_json(cli_schema_path)
    cli_schema.setdefault("arguments", [])
    cli_schema["arguments"].extend(
        copy.deepcopy(_GLOBAL_CLI_SCHEMA.get("arguments", []))
    )
    parser, arg_defs = build_cli_parser(cli_schema)
    args = parser.parse_args(argv)

    defaults_path = project_dir / _PROJECT_DEFAULTS_REL
    if not defaults_path.exists():
        raise FileNotFoundError(f"Missing defaults config: {defaults_path}")
    project_defaults = load_config(defaults_path)
    cfg = deep_update(copy.deepcopy(_GLOBAL_DEFAULTS), project_defaults)
    cfg = apply_cli_overrides(cfg, args, arg_defs, project_dir, invocation_dir)
    ensure_no_placeholders(cfg)

    resolved_data_root = resolve_data_root(cfg.get("data_root"), project_dir)
    cfg["data_root"] = str(resolved_data_root)
    os.environ.setdefault(ENV_DATA_ROOT, str(resolved_data_root))

    seed_value = cfg.get("seed")
    if seed_value is not None:
        seed_everything(seed_value)  # type: ignore[arg-type]

    timestamp = dt.datetime.now().strftime(_DEFAULT_TIMESTAMP_FORMAT)
    run_name = _DEFAULT_RUN_PATTERN.format(timestamp=timestamp)

    base_out = Path(cfg.get("outdir", "outdir"))
    run_dir = base_out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_level = (
        cfg.get("logging", {}).get("level", "info")
        if isinstance(cfg.get("logging"), dict)
        else "info"
    )
    configure_logging(log_level, run_dir / "run.log")

    snapshot_path = run_dir / "config_snapshot.json"
    snapshot_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    runtime_cfg = _prepare_runner_config(cfg)

    _log_run_banner(project_dir, cfg)
    runner_fn = import_callable(_DEFAULT_RUNNER_CALLABLE)
    runner_fn(runtime_cfg, run_dir)

    logging.info("Finished. Artifacts in: %s", run_dir)
    return run_dir


__all__ = ["run_from_project"]
