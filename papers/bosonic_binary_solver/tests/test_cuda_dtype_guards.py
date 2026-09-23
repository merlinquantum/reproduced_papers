"""Guards for CUDA dtype coverage gaps that CPU testing cannot see.

`torch.einsum` dispatches to `baddbmm`, which has no integer CUDA kernel: an
int64 einsum runs perfectly on CPU and dies on a GPU with

    NotImplementedError: "baddbmm_cuda" not implemented for 'Long'

That is not something a CPU test suite discovers by running, so it is discovered
here by construction: `torch.einsum` is temporarily replaced with a version that
refuses integer operands, exactly as CUDA does, and every batched cost function
is exercised through it.
"""

import numpy as np
import pytest
import torch
from lib.problems_torch import build_batch

INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)


@pytest.fixture
def cuda_like_einsum(monkeypatch):
    """Make torch.einsum reject integer operands, as the CUDA kernel does."""
    real = torch.einsum

    def guarded(equation, *operands):
        for operand in operands:
            if isinstance(operand, torch.Tensor) and operand.dtype in INTEGER_DTYPES:
                raise NotImplementedError(
                    f'"baddbmm_cuda" not implemented for {operand.dtype} '
                    f"(simulated: einsum on integer tensors fails on CUDA)"
                )
        return real(equation, *operands)

    monkeypatch.setattr(torch, "einsum", guarded)


@pytest.mark.parametrize(
    "family,m", [("knapsack", 30), ("tsp", 29), ("tsp", 19), ("tsp", 10)]
)
def test_batched_costs_avoid_integer_einsum(cuda_like_einsum, family, m):
    seeds = [0, 1]
    cost_batch, _ = build_batch(family, m, seeds, torch.device("cpu"))
    bits = torch.tensor(
        np.random.default_rng(0).integers(0, 2, (2, 64, m)), dtype=torch.int8
    )
    costs = cost_batch(bits)
    assert costs.shape == (2, 64)
    assert torch.isfinite(costs).all()


def test_tsp_index_stays_exact_at_the_largest_size():
    """The tour index must be computed in int64, not through float.

    At m=29 the index reaches 2^29; a float32 path would round it and silently
    decode the wrong tour, which no assertion on shapes would catch.
    """
    seeds = [0]
    cost_batch, _ = build_batch("tsp", 29, seeds, torch.device("cpu"))
    bits = torch.ones(1, 1, 29, dtype=torch.int8)  # index = 2^29 - 1
    powers = cost_batch.powers
    assert powers.dtype == torch.int64
    index = (bits.to(torch.int64) * powers).sum(dim=-1)
    assert int(index[0, 0]) == 2**29 - 1
