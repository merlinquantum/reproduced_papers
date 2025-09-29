import sys
from pathlib import Path

import torch

# Ensure qSSL/ is on path so `lib` is importable
THIS_DIR = Path(__file__).resolve().parent
QSSL_DIR = THIS_DIR.parent
if str(QSSL_DIR) not in sys.path:
    sys.path.insert(0, str(QSSL_DIR))

from lib.training_utils import InfoNCELoss  # noqa: E402


def test_infonce_loss_basic_properties():
    torch.manual_seed(0)
    criterion = InfoNCELoss(temperature=0.07)

    # Two batches of embeddings (B=4, D=8)
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)

    loss = criterion(z1, z2)

    # Loss returns a [1]-shaped tensor, finite and non-negative
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert torch.isfinite(loss).item() is True
    assert (loss >= 0).item() is True


def test_infonce_loss_identical_views_lower():
    torch.manual_seed(0)
    criterion = InfoNCELoss(temperature=0.07)

    # Make z2 close to z1; loss should be lower than with random
    z1 = torch.randn(8, 16)
    z2_close = z1 + 0.01 * torch.randn_like(z1)
    z2_rand = torch.randn(8, 16)

    loss_close = criterion(z1, z2_close)
    loss_rand = criterion(z1, z2_rand)

    assert loss_close.item() <= loss_rand.item() + 1e-5
