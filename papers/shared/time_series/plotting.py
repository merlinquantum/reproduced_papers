"""Small shared matplotlib helpers (headless-friendly)."""

from __future__ import annotations

from pathlib import Path


def maybe_use_agg_backend(*, show: bool) -> None:
    """Use a non-interactive backend unless the user requested --show.

    Must be called before importing matplotlib.pyplot.
    """

    if show:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)


def save_figure(plt, out_path: Path, *, dpi: int = 150) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    return out_path
