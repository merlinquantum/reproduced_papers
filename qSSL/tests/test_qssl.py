import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Ensure qSSL/lib is importable as `lib` (prepend qSSL/ to sys.path)
THIS_DIR = Path(__file__).resolve().parent
QSSL_DIR = THIS_DIR.parent
if str(QSSL_DIR) not in sys.path:
    sys.path.insert(0, str(QSSL_DIR))

from lib.model import QSSL  # noqa: E402


def _make_base_args():
    # Common defaults
    return {
        "width": 4,
        "batch_norm": False,
        "layers": 1,
        "loss_dim": 8,
        "temperature": 0.07,
        # Qiskit/MerLin optional fields
        "modes": 4,
        "no_bunching": False,
        "encoding": "vector",
        "q_ansatz": "sim_circ_14_half",
        "q_sweeps": 1,
        "activation": "null",
        "shots": 10,
        "q_backend": "qasm_simulator",
        "save_dhs": False,
    }


def _make_args_classical():
    base = _make_base_args()
    return SimpleNamespace(merlin=False, qiskit=False, **base)


def _make_args_qiskit():
    base = _make_base_args()
    return SimpleNamespace(merlin=False, qiskit=True, **base)


def _make_args_merlin():
    base = _make_base_args()
    return SimpleNamespace(merlin=True, qiskit=False, **base)


def test_qssl_forward_classical_smoke():
    args = _make_args_classical()
    model = QSSL(args)
    model.eval()

    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        loss = model(x1, x2)

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert torch.isfinite(loss).item() is True


def test_qssl_init_qiskit_smoke():
    # Only construct and run a tiny forward pass to keep it quick
    args = _make_args_qiskit()
    model = QSSL(args)
    model.eval()

    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        loss = model(x1, x2)

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert torch.isfinite(loss).item() is True


def test_qssl_init_merlin_smoke():
    # Construct MerLin model; forward pass might be heavier, keep batch=1
    args = _make_args_merlin()
    model = QSSL(args)
    model.eval()

    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        loss = model(x1, x2)

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert torch.isfinite(loss).item() is True
