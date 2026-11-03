"""Finite Gaussian reference pools for a matched precomputed-prior workflow."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import replace_file, sha256, write_json
from .pool_stream import PoolStream


def generate_classical_pool(source: str, settings: dict, folder: Path) -> dict:
    """Generate or verify independent, standardized classical latent vectors.

    Parameters
    ----------
    source : str
        ``gaussian`` for independent N(0,1) coordinates.
    settings : dict
        CPU device, sample count, chunk size, and event RNG seed.
    folder : Path
        Destination for a completed float32 pool with 32 coordinates.

    Returns
    -------
    dict
        Pool identity, hash, theoretical and measured moments, and timings.
    """
    if source != "gaussian":
        raise ValueError("Classical source must be gaussian")
    if settings["device"] != "cpu":
        raise ValueError("Precomputed classical pool generation uses CPU")
    samples, chunk = settings["samples"], settings["chunk_size"]
    if type(samples) is not int or samples < 1 or type(chunk) is not int or chunk < 1:
        raise ValueError("Samples and chunk_size must be positive integers")
    identity = {
        "source": source,
        "samples": samples,
        "latent_dim": 32,
        "sampling_seed": settings["seed"],
        "chunk_size": chunk,
        "device": "cpu",
        "dtype": "float32",
        "rng": "NumPy PCG64",
        "numpy_version": np.__version__,
    }
    complete = folder / "complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved["identity"] != identity or saved["samples_sha256"] != sha256(
            folder / "samples.npy"
        ):
            raise ValueError("Existing pool configuration or sample hash differs")
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / "samples.partial.npy"
    if partial.exists() or (folder / "samples.npy").exists():
        raise FileExistsError("Incomplete pool exists; use a fresh output directory")
    write_json(folder / "config.json", identity)
    values = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(samples, 32)
    )
    rng = np.random.Generator(np.random.PCG64(settings["seed"]))
    sums, products = np.zeros(32), np.zeros((32, 32))
    started = time.perf_counter()
    try:
        for offset in range(0, samples, chunk):
            size = min(chunk, samples - offset)
            block = rng.standard_normal((size, 32), dtype=np.float32)
            values[offset : offset + size] = block
            precise = block.astype(np.float64)
            sums += precise.sum(axis=0)
            products += precise.T @ precise
        values.flush()
    finally:
        del values
    replace_file(partial, folder / "samples.npy")
    mean = sums / samples
    covariance = products / samples - np.outer(mean, mean)
    completed = {
        "identity": identity,
        "status": "complete",
        "distribution": "Independent N(0,1)",
        "normalization": "Theoretical mean 0 and variance 1; no empirical/batch standardization",
        "expected_mean": [0.0] * 32,
        "expected_variance": [1.0] * 32,
        "observed_mean": mean.tolist(),
        "observed_variance": np.diag(covariance).tolist(),
        "max_abs_mean": float(abs(mean).max()),
        "max_variance_error": float(abs(np.diag(covariance) - 1).max()),
        "max_off_diagonal_covariance": float(
            abs(covariance - np.diag(np.diag(covariance))).max()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "samples_sha256": sha256(folder / "samples.npy"),
        "comparison_scope": "Finite precomputed pool; differs from fresh draws at every GAN step",
    }
    write_json(complete, completed)
    return completed


class ClassicalPoolPrior(PoolStream):
    """Load a verified finite pool, using the same event ordering as Fock pools.

    Parameters
    ----------
    folder : str | Path
        Directory containing samples.npy and complete.json.
    source : str
        Expected classical distribution, checked against pool metadata.
    sampler_seed : int
        Pool ordering seed, independent of generation and GAN seeds. Default 0.
    """

    def __init__(self, folder: str | Path, source: str, sampler_seed: int = 0):
        if source != "gaussian":
            raise ValueError("Classical source must be gaussian")
        folder = Path(folder)
        complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        identity = complete["identity"]
        path = folder / "samples.npy"
        if (
            complete["status"] != "complete"
            or identity["source"] != source
            or sha256(path) != complete["samples_sha256"]
        ):
            raise ValueError("Pool source, completion, or hash is invalid")
        values = np.load(path, allow_pickle=False)
        if (
            values.shape != (identity["samples"], 32)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Expected finite float32 latent vectors with shape [samples,32]"
            )
        self.values = torch.from_numpy(values)
        self.condition = source
        self.sampler_seed = sampler_seed
        self.order = np.arange(len(values))
        self.rng = np.random.RandomState(sampler_seed)
        self.rng.shuffle(self.order)
        self.window = int(np.rint(len(values) * 0.5))
        self.position = 0
        self.metadata = {
            "source": source,
            "identity": identity,
            "samples_sha256": complete["samples_sha256"],
            "normalization": "Theoretical zero mean/unit variance; generator RMSNorm retained",
            "sampler_seed": sampler_seed,
            "sampler": "Upstream InfiniteSampler window_size=0.5",
        }
