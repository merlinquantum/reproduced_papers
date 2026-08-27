import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.lib.qiskit_models import qiskit_QCNN  # noqa: E402


def test__preprocess_input():
    model = qiskit_QCNN()

    assert (
        torch.Size((1, 2**10)) == model._preprocess_input(torch.rand(1, 32, 32)).shape
    )
    assert torch.Size((1, 2**10)) == model._preprocess_input(torch.rand(1, 2**10)).shape
