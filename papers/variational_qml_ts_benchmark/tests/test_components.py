from __future__ import annotations

import torch
from lib.data import DataHandling, dataset_dim
from lib.models import build_model, count_parameters

DATA_ROOT = None  # resolved lazily below


def _root():
    from common import PROJECT_DIR

    return str(PROJECT_DIR.parent.parent / "data" / "variational_qml_ts_benchmark")


def test_dataset_dims():
    assert dataset_dim("mackey_1000") == 1
    assert dataset_dim("henon_1000") == 2
    assert dataset_dim("lorenz_1000") == 3


def test_data_shapes_and_range():
    dh = DataHandling("henon_1000", seq_length=4, prediction_step=1, data_root=_root())
    xtr, ytr, xval, yval, xte, yte = dh.get_training_and_test_data()
    assert xtr.shape[1:] == (4, 2)
    assert ytr.shape[1] == 2
    # min-max scaled to [0,1]
    assert float(xtr.min()) >= -1e-6 and float(xtr.max()) <= 1 + 1e-6


def test_ruqnn_bugfix_changes_encoding():
    ru = "ruexp_EYX_EZY_CX_CY_X_CZ_X_CZ_EXY_EXX_EZZ_X_EYZ_EXY_EZX_Y_EYX_CY_X_CY"
    x = torch.rand(3, 4, 2)
    m_orig = build_model("vqc", "henon_1000", 4, ru, 4, None, random_id=0, bugfix=False)
    m_fix = build_model("vqc", "henon_1000", 4, ru, 4, None, random_id=0, bugfix=True)
    m_orig.eval()
    m_fix.eval()
    # Same trainable params (seeded identically) but different encoding -> outputs differ.
    with torch.no_grad():
        assert not torch.allclose(m_orig(x), m_fix(x), atol=1e-5)


def test_all_models_build_and_forward():
    specs = [
        ("mlp", "relu_8", None, None),
        ("rnn", "layers_1", None, 8),
        ("lstm", "layers_1", None, 8),
        ("vqc", "paper_rivera-ruiz_with_inputlayer_1", 4, None),
        ("qrnn", "paper_no_reset", 4, 2),
        ("qlstm", "original_1", 4, None),
        ("le_qlstm", "original_1", 6, 8),
    ]
    x = torch.rand(2, 4, 2)  # henon (2-D)
    for name, ansatz, nq, hs in specs:
        m = build_model(name, "henon_1000", 4, ansatz, nq, hs, random_id=0, bugfix=True)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 2), name
        assert count_parameters(m) > 0, name


# --- Photonic reservoir extension (lib/reservoir.py) ------------------------
# These models are non-variational and sit outside the paper's scope; the tests
# below pin the properties that make them *reservoirs* and that make the
# memristive variant genuinely time-dependent.


def _reservoirs():
    return [
        build_model("photonic_reservoir", "henon_1000", 4, "reservoir", 6, 3, 0),
        build_model("photonic_memristor", "henon_1000", 4, "reservoir", 6, 3, 0),
    ]


def test_reservoirs_build_and_forward_on_every_system():
    for ds, d in [("mackey_1000", 1), ("henon_1000", 2), ("lorenz_1000", 3)]:
        x = torch.rand(3, 4, d)
        for name in ("photonic_reservoir", "photonic_memristor"):
            m = build_model(name, ds, 4, "reservoir", 6, 3, random_id=0)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape == (3, d), (name, ds)


def test_reservoir_circuit_is_frozen_only_readout_trains():
    """The defining reservoir property: no trainable circuit parameters."""
    for m in _reservoirs():
        assert not any(p.requires_grad for p in m.qlayer.parameters())
        assert not any(p.requires_grad for p in m.proj.parameters())
        assert all(p.requires_grad for p in m.readout.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        readout = sum(p.numel() for p in m.readout.parameters())
        assert trainable == readout > 0
        assert m.n_frozen > 0


def test_memristive_reservoir_is_time_ordered():
    """The memristive chip must respond to the order of the window.

    A static reservoir sees the window as a flat vector; the memristive one
    integrates it step by step, so reversing time must change the output.
    """
    m = build_model("photonic_memristor", "henon_1000", 4, "reservoir", 6, 3, 0)
    m.eval()
    x = torch.rand(4, 4, 2)
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x.flip(dims=[1])), atol=1e-6)


def test_memristive_state_resets_between_batches():
    """Windows must be independent: repeated calls agree, and an item's
    prediction must not depend on the other items it is batched with."""
    m = build_model("photonic_memristor", "henon_1000", 4, "reservoir", 6, 3, 0)
    m.eval()
    x = torch.rand(5, 4, 2)
    with torch.no_grad():
        first, second = m(x), m(x)
        assert torch.allclose(first, second, atol=1e-8)  # reset() each forward
        solo = m(x[2:3])
        assert torch.allclose(solo, first[2:3], atol=1e-5)


def test_capacity_control_matches_memristive_readout_but_has_no_memory():
    """`photonic_seqreservoir` is the control for `photonic_memristor`.

    It must have an identically sized trainable readout (so a difference in
    accuracy cannot be attributed to capacity) while carrying no optical memory.
    """
    mem = build_model("photonic_memristor", "henon_1000", 4, "reservoir", 6, 3, 0)
    ctl = build_model("photonic_seqreservoir", "henon_1000", 4, "reservoir", 6, 3, 0)
    n_mem = sum(p.numel() for p in mem.parameters() if p.requires_grad)
    n_ctl = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
    assert n_mem == n_ctl > 0
    assert mem.has_memory and not ctl.has_memory
    # Both remain deterministic across repeated evaluation.
    ctl.eval()
    x = torch.rand(3, 4, 2)
    with torch.no_grad():
        assert torch.allclose(ctl(x), ctl(x), atol=1e-8)


def test_reservoir_ansatz_parsing_and_defaults():
    """Hyperparameters ride on the ansatz string (repo convention).

    A bare `reservoir` ansatz must reproduce the original a-priori configuration
    exactly, so results produced before the search remain valid.
    """
    from lib.reservoir import parse_reservoir_ansatz

    assert parse_reservoir_ansatz("reservoir") == {}
    assert parse_reservoir_ansatz("reservoir_scale1.57_leak0.9_mem3") == {
        "scale": 1.57,
        "leak": 0.9,
        "mem": 3,
    }
    bare = build_model("photonic_memristor", "henon_1000", 4, "reservoir", 6, 3, 0)
    explicit = build_model(
        "photonic_memristor",
        "henon_1000",
        4,
        "reservoir_scale3.141592653589793_leak0.50_mem2",
        6,
        3,
        0,
    )
    bare.eval()
    explicit.eval()
    x = torch.rand(3, 4, 2)
    with torch.no_grad():
        assert torch.allclose(bare(x), explicit(x), atol=1e-8)


def test_memristor_drive_is_input_dependent_for_any_count():
    """The update rule must never collapse to a constant.

    With a stride of 1 the drive would sum the whole normalised distribution to
    exactly 1.0 for every input, making the memristor a fixed clock that carries
    no information. The stride is floored at 2 to prevent that.
    """
    for mem in (1, 2, 3):
        m = build_model(
            "photonic_memristor", "henon_1000", 4, f"reservoir_mem{mem}", 6, 3, 0
        )
        probs = torch.rand(4, m.qlayer.output_size)
        probs = probs / probs.sum(-1, keepdim=True)
        for k in range(mem):
            drive = m._make_update_rule(k)(torch.zeros(4), probs)
            assert drive.std() > 1e-6, (mem, k)
