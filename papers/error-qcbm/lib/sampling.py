"""Project-local exact count accumulation adapter for MerLin multinomial draws."""

from contextlib import contextmanager
from inspect import signature
from threading import RLock
from unittest.mock import patch

import torch
from merlin.measurement.process import SamplingProcess

_SAMPLER_LOCK = RLock()
_ORIGINAL_SAMPLER = SamplingProcess.pcvl_sampler


def vectorized_multinomial(self, distribution, shots, method=None):
    """Keep MerLin's Torch categorical draws and vectorize integer accumulation.

    Parameters
    ----------
    distribution : torch.Tensor
        One distribution or a row batch supplied by MerLin after detection.
    shots : int
        Native draws for each row.
    method : str or None
        Only multinomial is supported. Default value is None, using self.method.

    Returns
    -------
    torch.Tensor
        Frequencies with the original shape, dtype and device.
    """
    if shots <= 0:
        return distribution
    if (method or self.method) != "multinomial":
        raise ValueError("Vector count adapter supports only native multinomial draws")
    if distribution.ndim not in (1, 2):
        raise ValueError("Expected a 1D distribution or 2D distribution batch")
    rows = distribution.unsqueeze(0) if distribution.ndim == 1 else distribution
    outputs = []
    for row in rows:
        indices = torch.multinomial(row, num_samples=shots, replacement=True)
        counts = torch.bincount(indices, minlength=row.numel()).to(row.dtype)
        outputs.append(counts / shots)
    return outputs[0] if distribution.ndim == 1 else torch.stack(outputs)


@contextmanager
def count_accumulation(mode):
    """Scope the adapter to one forward; never edit installed MerLin files.

    Parameters
    ----------
    mode : str
        python for stock MerLin, or vectorized for the exact local adapter.

    Yields
    ------
    None
        Project backend forwards share a lock while the process-local hook lives.
    """
    if mode not in ("python", "vectorized"):
        raise ValueError("Count accumulation must be python or vectorized")
    with _SAMPLER_LOCK:
        if SamplingProcess.pcvl_sampler is not _ORIGINAL_SAMPLER:
            raise RuntimeError("MerLin sampler already patched outside this adapter")
        if tuple(signature(_ORIGINAL_SAMPLER).parameters) != (
            "self",
            "distribution",
            "shots",
            "method",
        ):
            raise RuntimeError("Unsupported MerLin pcvl_sampler signature")
        if mode == "python":
            yield
        else:
            with patch.object(SamplingProcess, "pcvl_sampler", vectorized_multinomial):
                yield
