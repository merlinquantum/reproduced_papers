"""Fock-specific standardization with checkpointable upstream pool ordering."""

import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256
from .pool_stream import PoolStream


class FockPoolPrior(PoolStream):
    """Load a verified Fock pool; leave the generator's RMSNorm unchanged.

    Parameters
    ----------
    folder : str | Path
        Completed sample directory with sample and moment files.
    sampler_seed : int
        InfiniteSampler ordering seed. Default is 0, as upstream.
    """

    def __init__(self, folder: str | Path, sampler_seed: int = 0):
        folder = Path(folder)
        complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        identity = complete["identity"]
        condition = identity["condition"]
        if condition not in {"dist", "indist"}:
            raise ValueError("Photonic pool condition must be dist or indist")
        samples_file = folder / "samples.npy"
        if (
            complete["status"] != "complete"
            or sha256(samples_file) != complete["samples_sha256"]
        ):
            raise ValueError("Pool is incomplete or its sample hash differs")
        if sha256(folder / "moments.npz") != complete["moments_sha256"]:
            raise ValueError("Pool normalization moments changed")
        counts = np.load(samples_file, mmap_mode="r", allow_pickle=False)
        if counts.shape != (identity["samples"], 32) or not np.issubdtype(
            counts.dtype, np.integer
        ):
            raise ValueError("GAN requires an integer pool with 32 modes")
        with np.load(folder / "moments.npz", allow_pickle=False) as moments:
            self.mean = moments["expected_mean"]
            variance = np.diag(moments["expected_covariance"])
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError(
                "Fock normalization needs a positive variance in every mode"
            )
        self.std = np.sqrt(variance)
        values = np.empty(counts.shape, dtype=np.float32)
        for offset in range(0, len(counts), 100000):
            block = counts[offset : offset + 100000]
            if np.any(block < 0) or not np.all(
                block.sum(axis=1) == identity["photons"]
            ):
                raise ValueError("Invalid Fock counts")
            values[offset : offset + len(block)] = (
                block.astype(np.float64) - self.mean
            ) / self.std
        if not np.isfinite(values).all():
            raise ValueError("Fock normalization produced nonfinite values")
        self.values = torch.from_numpy(values)
        self.condition = condition
        self.sampler_seed = sampler_seed
        self.order = np.arange(len(values))
        self.rng = np.random.RandomState(sampler_seed)
        self.rng.shuffle(self.order)
        self.window = int(np.rint(len(values) * 0.5))
        self.position = 0
        self.metadata = {
            "source": "fock",
            "identity": identity,
            "samples_sha256": complete["samples_sha256"],
            "moments_sha256": sha256(folder / "moments.npz"),
            "normalization": f"Ideal {condition} Fock per-mode analytic mean/std; float64 then float32",
            "normalization_mean": self.mean.tolist(),
            "normalization_std": self.std.tolist(),
            "sampler_seed": sampler_seed,
            "sampler": "Upstream InfiniteSampler window_size=0.5",
        }
