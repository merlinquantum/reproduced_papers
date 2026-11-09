"""The GPU path must not depend on Perceval or MerLin.

The GPU host installs torch, triton and numpy. An import chain that quietly
reaches lib.sampler (Perceval) or lib.bbs_merlin (MerLin) makes a queue fail at
launch on a machine where those are absent -- which is exactly what happened the
first time this queue was launched.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import sys
sys.path.insert(0, {root!r})
for name in ("perceval", "merlin"):
    sys.modules[name] = None          # any real import of these now fails loudly
import lib.bbs_gpu, lib.problems_torch, lib.sampler_torch, lib.metrics, lib.tbi
import importlib.util
spec = importlib.util.spec_from_file_location("run_gpu", {runner!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("clean")
"""


def test_gpu_modules_import_without_perceval_or_merlin():
    code = PROBE.format(root=str(ROOT), runner=str(ROOT / "utils" / "run_gpu.py"))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert "clean" in result.stdout, result.stderr
