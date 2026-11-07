"""Explicit device selection for sampling and optional GAN training."""

import torch


def checked_device(name: str, backend: str | None = None) -> torch.device:
    """Validate a requested CPU or CUDA device without silent fallback.

    Parameters
    ----------
    name : str
        ``cpu``, ``cuda``, or an indexed CUDA device.
    backend : str | None
        Sampling backend, when applicable. Default is None.

    Returns
    -------
    torch.device
        Available device supported by the selected implementation.
    """
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    if backend == "perceval" and device.type != "cpu":
        raise ValueError("Perceval Clifford2017 uses CPU; select sampling.device=cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Install CUDA-enabled PyTorch and check the GPU."
            )
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
    return device
