"""Phase 04 M2 check (run on a CUDA machine): the MerLin engine's Clifford sampler is seeded and exact.

Usage::

    python papers/photonic_MMD_generative/utils/check_merlin_sampler.py

Writes ``outdir/merlin_sampler_check.json`` with: bitwise equality of two draws
with the same seed, inequality for a different seed, photon-number
conservation, and the per-mode mean occupation of 20 000 samples against the
exact single-mode occupation probabilities (App. B.2 formula, collision-free
part ignored only through the < 1 photon-per-mode assumption — reported as the
max z-score of the binarized occupation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))

from lib import circuits, initialization, qcbm_merlin  # noqa: E402


def main() -> None:
    """Run the checks on a random m=16, n=4 interferometer."""
    m, modes, shots = 16, [2, 5, 8, 11], 20000
    U = np.asarray(
        circuits.haar_unitary(jax.random.normal(jax.random.PRNGKey(0), (2 * m * m,)), m)
    )
    a = qcbm_merlin.merlin_clifford_samples(U, modes, shots, 7, "cuda")
    b = qcbm_merlin.merlin_clifford_samples(U, modes, shots, 7, "cuda")
    c = qcbm_merlin.merlin_clifford_samples(U, modes, shots, 8, "cuda")
    occupied = np.asarray(
        initialization.model_marginals(
            jnp.abs(jnp.asarray(U)[:, jnp.asarray(modes)]) ** 2
        )
    )
    observed = (a > 0).mean(axis=0)
    z = (observed - occupied) / np.sqrt(
        np.maximum(occupied * (1 - occupied), 1e-12) / shots
    )
    result = {
        "same_seed_bitwise_equal": bool(np.array_equal(a, b)),
        "different_seed_differs": bool(not np.array_equal(a, c)),
        "photon_number_conserved": bool((a.sum(axis=1) == len(modes)).all()),
        "max_abs_z_occupation": float(np.max(np.abs(z))),
        "shots": shots,
        "m": m,
        "input_modes": modes,
    }
    out = PAPER_DIR / "outdir" / "merlin_sampler_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
