"""Execute ``notebook.ipynb`` top to bottom in a fresh kernel and save its outputs.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/run_notebook.py
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, NOTEBOOK)
    counts = [c.get("execution_count") for c in notebook.cells if c.cell_type == "code"]
    print(
        f"executed {len(counts)} code cells in {time.perf_counter() - started:.0f} s; sequential: {counts == list(range(1, len(counts) + 1))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
