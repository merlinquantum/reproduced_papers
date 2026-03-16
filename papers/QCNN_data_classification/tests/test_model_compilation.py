"""Lightweight model compilation checks."""

import torch


def test_compile_classical_qconv():
    from lib.models import QConvModel

    model = QConvModel(input_dim=4, n_kernels=1, kernel_size=2, stride=1, bias=True)
    out = model(torch.zeros(1, 4))
    assert out.shape == (1, 2)


def test_compile_quantum_qconv():
    from lib.models import QConvModel, build_quantum_kernels

    kernels = build_quantum_kernels(
        n_kernels=1,
        kernel_size=2,
        kernel_modes=4,
        n_photons=1,
        state_pattern="default",
        reservoir_mode=False,
        amplitudes_encoding=False,
    )
    model = QConvModel(
        input_dim=4,
        n_kernels=1,
        kernel_size=2,
        stride=1,
        bias=False,
        kernel_modules=kernels,
        amplitudes_encoding=False,
    )
    assert model.output_features > 0
