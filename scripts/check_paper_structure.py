#!/usr/bin/env python3
"""Validate paper directory structure against repository conventions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
SKIP_PAPERS = {"shared"}
REQUIRED_TOP_LEVEL = (
    "README.md",
    "requirements.txt",
    "configs",
    "cli.json",
    "lib",
    "tests",
)
WARNING_TOP_LEVEL = ("models", "results")
NOTEBOOK_NAMES = ("notebook.ipynb",)
LEGACY_NOTEBOOK_DIRS = ("notebooks",)
LEGACY_AUX_DIRS = ("scripts",)
REQUIRED_AUX_DIR = "utils"


@dataclass
class ValidationResult:
    paper: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, fail_on_warnings: bool) -> bool:
        if self.errors:
            return False
        if fail_on_warnings and self.warnings:
            return False
        return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate paper directory structure. "
            "Without positional paths, validates all papers."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Changed paths. The script infers impacted papers from files under "
            "papers/<name>/..."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every paper under papers/.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a non-zero status when warnings are found.",
    )
    return parser.parse_args(argv)


def iter_paper_dirs() -> list[Path]:
    paper_dirs: list[Path] = []
    for path in PAPERS_DIR.iterdir():
        if not path.is_dir() or path.name in SKIP_PAPERS:
            continue

        nested_projects = find_nested_projects(path)
        if nested_projects:
            paper_dirs.extend(nested_projects)
            continue

        paper_dirs.append(path)

    return sorted(paper_dirs)


def is_project_root(path: Path) -> bool:
    return (
        (path / "configs" / "defaults.json").is_file()
        and (path / "cli.json").is_file()
        and (path / "lib" / "runner.py").is_file()
    )


def find_nested_projects(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir() if child.is_dir() and is_project_root(child)
    )


def infer_papers(paths: list[str]) -> list[Path]:
    if not paths:
        return iter_paper_dirs()

    resolved: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        parts = path.parts
        if len(parts) < 2 or parts[0] != "papers":
            continue
        root_name = parts[1]
        if root_name in SKIP_PAPERS:
            continue

        top_level = PAPERS_DIR / root_name
        if not top_level.is_dir():
            continue

        nested_projects = find_nested_projects(top_level)
        if nested_projects and len(parts) >= 3:
            nested_candidate = top_level / parts[2]
            if nested_candidate in nested_projects:
                resolved.add(nested_candidate)
                continue

        if nested_projects:
            resolved.update(nested_projects)
            continue

        resolved.add(top_level)

    return sorted(resolved)


def has_notebook(paper_dir: Path) -> bool:
    if any((paper_dir / name).is_file() for name in NOTEBOOK_NAMES):
        return True
    if any((paper_dir / name).is_dir() for name in LEGACY_NOTEBOOK_DIRS):
        return True
    return any(paper_dir.glob("*.ipynb"))


def has_aux_dir(paper_dir: Path) -> bool:
    if (paper_dir / REQUIRED_AUX_DIR).is_dir():
        return True
    return any((paper_dir / name).is_dir() for name in LEGACY_AUX_DIRS)


def has_gitignore(paper_dir: Path) -> bool:
    if (paper_dir / ".gitignore").is_file():
        return True
    parent = paper_dir.parent
    return parent != PAPERS_DIR and (parent / ".gitignore").is_file()


def validate_paper(paper_dir: Path) -> ValidationResult:
    result = ValidationResult(paper=paper_dir.name)

    for entry in REQUIRED_TOP_LEVEL:
        candidate = paper_dir / entry
        if not candidate.exists():
            result.errors.append(f"missing required entry `{entry}`")

    for entry in WARNING_TOP_LEVEL:
        candidate = paper_dir / entry
        if not candidate.exists():
            result.warnings.append(f"missing recommended entry `{entry}`")

    if not has_notebook(paper_dir):
        result.errors.append(
            "missing notebook artifact (`notebook.ipynb`, another top-level "
            "`.ipynb`, or `notebooks/`)"
        )

    if not has_aux_dir(paper_dir):
        result.errors.append("missing `utils/` directory (or legacy `scripts/`)")

    defaults_path = paper_dir / "configs" / "defaults.json"
    if not defaults_path.is_file():
        result.errors.append("missing required file `configs/defaults.json`")

    runner_path = paper_dir / "lib" / "runner.py"
    if not runner_path.is_file():
        result.errors.append("missing required file `lib/runner.py`")

    if not has_gitignore(paper_dir):
        result.warnings.append(
            "missing `.gitignore` (locally or on the multi-part paper parent)"
        )

    for config_path in sorted((paper_dir / "configs").glob("*.json")):
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except json.JSONDecodeError as exc:
            result.errors.append(
                f"invalid JSON in `{config_path.relative_to(REPO_ROOT)}`: {exc}"
            )
            continue

    results_dir = paper_dir / "results"
    if results_dir.is_dir():
        for path in sorted(results_dir.rglob("run_*")):
            if path.is_dir():
                rel = path.relative_to(REPO_ROOT)
                result.errors.append(
                    f"generated run directory committed under `results/`: `{rel}`"
                )

    return result


def print_result(result: ValidationResult) -> None:
    if not result.errors and not result.warnings:
        print(f"[OK] {result.paper}")
        return

    print(f"[{result.paper}]")
    for issue in result.errors:
        print(f"  ERROR: {issue}")
    for issue in result.warnings:
        print(f"  WARN: {issue}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.all:
        paper_dirs = iter_paper_dirs()
    else:
        paper_dirs = infer_papers(args.paths)

    if not paper_dirs:
        return 0

    results = [validate_paper(paper_dir) for paper_dir in paper_dirs]
    for result in results:
        print_result(result)

    return 0 if all(result.ok(args.fail_on_warnings) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
