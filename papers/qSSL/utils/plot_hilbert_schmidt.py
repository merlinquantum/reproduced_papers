"""Regenerate a qSSL Hilbert-Schmidt tracking figure from saved metrics."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lib.training_utils import plot_loss_and_hilbert_schmidt  # noqa: E402

# Allow this utility to be invoked from the repository root or papers/qSSL.
QSSL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QSSL_DIR.parents[1]
for import_path in (REPOSITORY_ROOT, QSSL_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_hilbert_schmidt_metrics(metrics_path: Path) -> list[dict]:
    """Load per-batch Hilbert-Schmidt metrics from a JSON file.

    Parameters
    ----------
    metrics_path : pathlib.Path
        Path to a ``hilbert_schmidt_metrics.json`` file.

    Returns
    -------
    list[dict]
        Saved per-batch metric records.
    """
    with metrics_path.open() as metrics_file:
        return json.load(metrics_file)


def regenerate_plot(metrics_path: Path, output_path: Path, backend: str) -> None:
    """Regenerate a Hilbert-Schmidt tracking plot from saved metrics.

    Parameters
    ----------
    metrics_path : pathlib.Path
        Path to the saved metric history.
    output_path : pathlib.Path
        Destination path for the regenerated PNG figure.
    backend : str
        Backend name used in the figure title. Must be ``qiskit`` or
        ``merlin``.
    """
    dhs_history = load_hilbert_schmidt_metrics(metrics_path)
    args = SimpleNamespace(merlin=backend == "merlin")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_loss_and_hilbert_schmidt(
        [metric["loss"] for metric in dhs_history],
        dhs_history,
        args,
        str(output_path.parent),
    )

    generated_path = output_path.parent / "hilbert_schmidt_tracking.png"
    if generated_path != output_path:
        generated_path.replace(output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for plot regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate a qSSL Hilbert-Schmidt plot from saved JSON metrics."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("qiskit", "merlin"), required=True)
    return parser.parse_args()


def main() -> None:
    """Regenerate the requested figure."""
    arguments = parse_arguments()
    regenerate_plot(arguments.metrics, arguments.output, arguments.backend)


if __name__ == "__main__":
    main()
