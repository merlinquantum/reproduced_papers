from __future__ import annotations

from collections.abc import Mapping
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE_DIRS = {
    "DHO": PROJECT_ROOT / "lib" / "DHO" / "models",
    "SEE": PROJECT_ROOT / "lib" / "SEE" / "models",
    "DEE": PROJECT_ROOT / "lib" / "DEE" / "models",
    "TAF": PROJECT_ROOT / "lib" / "TAF" / "models",
}


def _should_sync(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix == ".pt"
        and not any(part.startswith(".") for part in path.parts)
    )


def sync_selected_models(
    *,
    models_root: str | Path | None = None,
    source_dirs: Mapping[str, str | Path] | None = None,
) -> list[Path]:
    mirrored_paths: list[Path] = []
    resolved_models_root = (
        Path(models_root) if models_root is not None else PROJECT_ROOT / "models"
    )
    resolved_source_dirs = {
        group_name: Path(source_dir)
        for group_name, source_dir in (source_dirs or MODEL_SOURCE_DIRS).items()
    }

    for group_name, source_dir in resolved_source_dirs.items():
        if not source_dir.is_dir():
            continue

        destination_dir = resolved_models_root / group_name
        for source in sorted(path for path in source_dir.rglob("*.pt") if _should_sync(path)):
            destination = destination_dir / source.relative_to(source_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            mirrored_paths.append(destination)
            print(f"Synced {source} -> {destination}")

    return mirrored_paths


if __name__ == "__main__":
    sync_selected_models()
