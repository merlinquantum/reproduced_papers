"""Execute notebook.ipynb top to bottom in a fresh kernel and save its outputs in place.

Usage (from the paper folder, in the `qelm_concentration` env)::

    python utils/execute_notebook.py --kernel qelm_concentration

Exits with status 1 when any cell raises or writes to stderr.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PAPER_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True, help="Jupyter kernel name")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per cell")
    args = parser.parse_args()

    path = PAPER_DIR / "notebook.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(PAPER_DIR)}},
    )
    start = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - start
    # Keep a generic kernelspec so the notebook opens in any Python 3 environment.
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(notebook, path)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    counts = [cell.execution_count for cell in code_cells]
    problems = [
        (index, output.get("ename") or output.get("name"))
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
        or (output.output_type == "stream" and output.name == "stderr")
    ]
    print(
        f"python {sys.version.split()[0]} | {platform.platform()} | {platform.processor()}"
    )
    print(
        f"code cells: {len(code_cells)} | sequential: {counts == list(range(1, len(counts) + 1))}"
    )
    print(f"errors or stderr outputs: {len(problems)} {problems}")
    print(f"elapsed: {elapsed:.0f} s")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
