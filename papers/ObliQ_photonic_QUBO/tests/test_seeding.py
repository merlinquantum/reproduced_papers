"""The seed-derivation contract: stable, decorrelated, and process-independent."""

import subprocess
import sys

from lib.seeding import SEED_MODULUS, derive_seed, set_global_seed


def test_derive_seed_is_deterministic():
    assert derive_seed(101200, "solver", "obliq-hybrid") == derive_seed(
        101200, "solver", "obliq-hybrid"
    )


def test_derive_seed_separates_labels_and_bases():
    assert derive_seed(101200, "a") != derive_seed(101200, "b")
    assert derive_seed(101200, "a") != derive_seed(101201, "a")


def test_derive_seed_is_in_range():
    for base in (0, 1, 101200, 2**40):
        assert 0 <= derive_seed(base, "solver") < SEED_MODULUS


def test_derive_seed_passes_none_through():
    """``None`` means "explicitly unseeded"; it must not become a number."""
    assert derive_seed(None, "solver", "obliq-hybrid") is None


def test_solver_sub_seeds_differ_per_solver():
    """Two solvers on the same instance must not share an initialization."""
    assert derive_seed(101200, "solver", "obliq-hybrid") != derive_seed(
        101200, "solver", "QAOA"
    )


def test_solver_and_baseline_sub_seeds_are_independent():
    """The solver seed and the beta baseline seed come off the same instance seed."""
    assert derive_seed(101200, "solver", "obliq-hybrid") != derive_seed(101200, "beta")


def test_derive_seed_is_stable_across_processes():
    """The sweep derives seeds inside spawned workers.

    Python salts ``hash()`` per process, so a hash-based derivation would give a
    worker a different sub-seed than the parent. This pins the blake2b behaviour
    by deriving in a *fresh interpreter* with a different PYTHONHASHSEED.
    """
    expected = derive_seed(101200, "solver", "obliq-hybrid")
    code = (
        f"import sys; sys.path.insert(0, {str(sys.path[0])!r});"
        "from lib.seeding import derive_seed;"
        "print(derive_seed(101200, 'solver', 'obliq-hybrid'))"
    )
    for hash_seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
        )
        assert int(out.stdout.strip()) == expected


def test_set_global_seed_makes_numpy_and_torch_repeatable():
    import numpy as np
    import torch

    set_global_seed(4242)
    first = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_global_seed(4242)
    second = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert first == second


def test_set_global_seed_ignores_none():
    """A ``None`` seed must leave the global generators alone, not reset them."""
    import numpy as np

    set_global_seed(11)
    before = np.random.rand()
    set_global_seed(None)
    assert np.random.rand() != before
