"""Execute ``notebook.ipynb`` top to bottom in a fresh kernel and record the evidence.

The notebook is executed in place with ``nbclient`` (kernel ``python3``,
working directory = this paper folder). Writes
``results/notebook_execution.json`` with the command, start time, elapsed
seconds, exit status, execution counts, error and stderr counts, and the
hardware. Exits non-zero if any cell fails.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/execute_notebook.py
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_DIR / "notebook.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    started = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    start = time.time()
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_DIR)}},
    )
    exit_status = 0
    try:
        client.execute()
    except CellExecutionError:
        exit_status = 1
    finally:
        nbformat.write(notebook, NOTEBOOK)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    counts = [cell.execution_count for cell in code_cells]
    record = {
        "command": ".venv/bin/python papers/QML_financial_benchmark/utils/execute_notebook.py",
        "kernel": "python3 (repository .venv), fresh kernel per execution",
        "started_utc": started,
        "elapsed_seconds": round(time.time() - start, 1),
        "exit_status": exit_status,
        "code_cells": len(code_cells),
        "execution_counts_sequential": counts == list(range(1, len(counts) + 1)),
        "error_outputs": sum(
            output.output_type == "error"
            for cell in code_cells
            for output in cell.outputs
        ),
        "stderr_outputs": sum(
            output.output_type == "stream" and output.name == "stderr"
            for cell in code_cells
            for output in cell.outputs
        ),
        "hardware": f"{platform.machine()} {platform.system()} {platform.release()}, CPU only (no CUDA/MPS used)",
        "python": platform.python_version(),
    }
    (PROJECT_DIR / "results" / "notebook_execution.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
