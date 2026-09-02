from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from lib.training import train
from torch.utils.data import DataLoader, TensorDataset


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_trainable_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"attention": 0, "total": total}


def _make_loader() -> DataLoader:
    x = torch.randn(32, 1, 2, 2)
    y = torch.randint(0, 2, (32, 1))
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_train_resume_continues_history(tmp_path: Path) -> None:
    loader = _make_loader()
    device = torch.device("cpu")

    cfg1 = {
        "epochs": 1,
        "lr": 1e-2,
        "dataset": "synthetic",
        "seed": 123,
        "model_type": "A",
        "circuit_family": "generic",
    }
    model1 = TinyNet()
    res1 = train(model1, loader, loader, loader, 2, cfg1, str(tmp_path), device)

    assert res1["last_completed_epoch"] == 1
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "progress.json").exists()

    cfg2 = dict(cfg1)
    cfg2["epochs"] = 2
    model2 = TinyNet()
    res2 = train(
        model2,
        loader,
        loader,
        loader,
        2,
        cfg2,
        str(tmp_path),
        device,
        resume_checkpoint=str(tmp_path / "last.pt"),
    )

    assert res2["resumed_from_checkpoint"] is True
    assert res2["last_completed_epoch"] == 2
    assert [entry["epoch"] for entry in res2["history"]] == [1, 2]
