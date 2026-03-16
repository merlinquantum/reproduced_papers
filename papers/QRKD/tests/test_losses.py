from __future__ import annotations

import torch
from lib.losses import DistillationLoss, fidelity_kernel_matrix


def test_fidelity_kernel_matrix_simple_backend():
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kernel = fidelity_kernel_matrix(feats, backend="simple")
    expected = torch.eye(2)
    assert torch.allclose(kernel, expected, atol=1e-6)


def test_distillation_loss_with_simple_kernel_components():
    torch.manual_seed(0)
    logits_s = torch.randn(4, 10)
    logits_t = torch.randn(4, 10)
    feat_s = torch.randn(4, 6)
    feat_t = torch.randn(4, 6)

    loss_fn = DistillationLoss(kd=0.7, dr=0.3, ar=0.2, qk=0.1, qk_backend="simple")
    loss = loss_fn(logits_s, logits_t, feat_s, feat_t)
    assert torch.isfinite(loss)
    assert loss.item() > 0
