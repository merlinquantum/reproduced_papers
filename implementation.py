#!/usr/bin/env python3
"""Repository-wide CLI entry point for reproduced papers."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from runtime_lib import run_from_project
from runtime_lib.cli import build_cli_parser

PAPERS_DIRNAME = "papers"
PROJECT_MARKERS = (
    Path("configs") / "defaults.json",
    Path("cli.json"),
    Path("lib") / "runner.py",
)
GLOBAL_CLI_SCHEMA = Path("runtime_lib") / "global_cli.json"


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_project_parser(
    project_dir: Path, repo_root: Path
) -> argparse.ArgumentParser:
    cli_schema_path = project_dir / "cli.json"
    cli_schema = _load_json(cli_schema_path)
    cli_schema.setdefault("arguments", [])
    global_schema_path = repo_root / GLOBAL_CLI_SCHEMA
    if global_schema_path.exists():
        global_schema = _load_json(global_schema_path)
        cli_schema["arguments"].extend(
            copy.deepcopy(global_schema.get("arguments", []))
        )
    parser, _ = build_cli_parser(cli_schema)
    return parser


def _format_help_without_usage(parser: argparse.ArgumentParser) -> str:
    help_text = parser.format_help()
    lines = help_text.splitlines()
    if lines and lines[0].lower().startswith("usage:"):
        lines = lines[1:]
    return "\n".join(lines).rstrip()


def _is_project_dir(path: Path) -> bool:
    path = path.resolve()
    if not path.is_dir():
        return False
    return all((path / marker).exists() for marker in PROJECT_MARKERS)


def _iter_project_dirs(repo_root: Path) -> list[Path]:
    """Return all project dirs (configs/defaults.json, cli.json, lib/runner.py).

    Searches recursively under `papers/` (to capture nested subprojects such as
    fock_state_expressivity/*) and also at the repository root.
    """

    bases = [repo_root / PAPERS_DIRNAME, repo_root]
    seen: set[Path] = set()
    found: list[Path] = []

    for base in bases:
        if not base.exists():
            continue
        # Find any configs/defaults.json and test its parent as a project dir.
        for defaults_path in base.rglob("configs/defaults.json"):
            project_dir = defaults_path.parent.parent
            if project_dir in seen:
                continue
            if _is_project_dir(project_dir):
                found.append(project_dir)
                seen.add(project_dir)

    return sorted(found, key=lambda p: p.as_posix())


def _find_enclosing_project_dir(start: Path, repo_root: Path) -> Path | None:
    start = start.resolve()
    for path in [start, *start.parents]:
        if _is_project_dir(path):
            return path
        if path == repo_root:
            break
    return None


def _resolve_named_project(repo_root: Path, name: str) -> Path | None:
    raw = Path(name).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                repo_root / raw,
                repo_root / PAPERS_DIRNAME / raw,
                repo_root / raw.name,
                repo_root / PAPERS_DIRNAME / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _require_project_dir(
    parser: argparse.ArgumentParser, project_dir: Path | None
) -> Path:
    if project_dir is None:
        parser.error(
            "Cannot infer paper directory. Specify --paper/--paper-name/--paper-dir "
            "or run the command from inside a paper directory."
        )
    if not _is_project_dir(project_dir):
        parser.error(
            "Invalid paper directory (expected configs/defaults.json, cli.json, "
            f"and lib/runner.py): {project_dir}"
        )
    return project_dir.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show help (includes paper options when available)",
    )
    parser.add_argument(
        "--paper",
        "--paper-name",
        "--paper_name",
        "-p",
        dest="paper",
        help="Target paper folder under papers/ (e.g., QLSTM, QORC)",
    )
    parser.add_argument(
        "--paper-dir",
        "--paper_dir",
        dest="paper_dir",
        default=None,
        help="Explicit path to a paper directory (overrides --paper)",
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        dest="list_papers",
        help=(
            "List available paper folders that contain configs/defaults.json, "
            "cli.json, and lib/runner.py"
        ),
    )
    parser.add_argument(
        "--project",
        "--project-name",
        "--project_name",
        dest="paper",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--project-dir",
        "--project_dir",
        dest="paper_dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        dest="list_papers",
        help=argparse.SUPPRESS,
    )
    known, remaining = parser.parse_known_args(argv)

    repo_root = _find_repo_root()
    if known.show_help:
        project_dir: Path | None
        if known.paper:
            project_dir = _resolve_named_project(repo_root, known.paper)
            if project_dir is None:
                print(
                    f"Error: unknown paper '{known.paper}'. Use --list-papers to list available papers.",
                    file=sys.stderr,
                )
                return 2
        elif known.paper_dir:
            project_dir = Path(known.paper_dir).expanduser().resolve()
            if not project_dir.exists():
                print(
                    f"Error: paper directory does not exist: {project_dir}",
                    file=sys.stderr,
                )
                return 2
        else:
            project_dir = _find_enclosing_project_dir(Path.cwd(), repo_root)
        if project_dir and not _is_project_dir(project_dir):
            print(
                "Error: invalid paper directory (expected configs/defaults.json, cli.json, "
                f"and lib/runner.py): {project_dir}",
                file=sys.stderr,
            )
            return 2
        if project_dir and _is_project_dir(project_dir):
            project_parser = _build_project_parser(project_dir, repo_root)
            prog = Path(sys.argv[0]).name
            paper_name = project_dir.name
            print(f"Options for paper {paper_name} (includes global options).")
            print(f"usage: {prog} --paper {paper_name} [options]")
            help_body = _format_help_without_usage(project_parser)
            if help_body:
                print(help_body)
        else:
            parser.print_help()
            print("\nTip: use --paper NAME --help to show paper-specific options.")
            print("\nTip: use --list-papers to list available papers.")
        return 0
    if known.list_papers:
        for path in _iter_project_dirs(repo_root):
            print(path.relative_to(repo_root))
        return 0

    project_dir: Path | None
    if known.paper:
        project_dir = _resolve_named_project(repo_root, known.paper)
    elif known.paper_dir:
        project_dir = Path(known.paper_dir).expanduser().resolve()
    else:
        project_dir = _find_enclosing_project_dir(Path.cwd(), repo_root)

    project_dir = _require_project_dir(parser, project_dir)
    run_from_project(project_dir, remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
