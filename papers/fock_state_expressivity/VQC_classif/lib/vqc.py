"""
Variational Quantum Classifier (VQC) implementations for photonic circuits.

This module provides various quantum circuit architectures for classification tasks,
including beam splitter meshes, general interferometers, and specialized configurations.
All circuits are designed to work with MerLin quantum machine learning framework.
"""

import random
from math import comb

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


def create_vqc_spiral(m, input_size, frequency=1):
    """
    Create quantum circuit with spiral architecture for classification.

    Args:
        m (int): Number of modes in the photonic circuit
        input_size (int): Number of input features to encode
        frequency (int): Number of encoding-processing layers

    Returns:
        pcvl.Circuit: The constructed spiral quantum circuit

    Note:
        Based on spiral dataset classification from Quandela's MerLin notebooks.
    """
    wl = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS(theta=pcvl.P(f"bs_1_{i}"))
            // pcvl.PS(pcvl.P(f"phase_1_{i}"))
            // pcvl.BS(theta=pcvl.P(f"bs_2_{i}"))
            // pcvl.PS(pcvl.P(f"phase_2_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    for f in range(frequency):
        c_var = pcvl.Circuit(m)
        for i in range(input_size):
            px = pcvl.P(f"px-{f}-{i + 1}")
            c_var.add(i % m, pcvl.PS(px))

        c.add(0, c_var, merge=True)
        wr = pcvl.GenericInterferometer(
            m,
            lambda i: (
                pcvl.BS()
                // pcvl.PS(pcvl.P(f"phase_3_{i}"))
                // pcvl.BS()
                // pcvl.PS(pcvl.P(f"phase_4_{i}"))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        c.add(0, wr, merge=True)

    return c


def create_vqc_bs_basic(m, input_size):
    """
    Create basic beam splitter based variational quantum classifier.

    Args:
        m (int): Number of modes in the photonic circuit (currently supports m=3)
        input_size (int): Number of input features to encode (currently supports input_size=2)

    Returns:
        pcvl.Circuit: The constructed basic beam splitter quantum circuit

    Note:
        Currently only supports 3 modes and 2 input features.
    """

    bs_l = pcvl.Circuit(m)
    bs_l.add(0, pcvl.BS(theta=pcvl.P("theta_l0")))
    bs_l.add(0, pcvl.PS(phi=pcvl.P("theta_l1")))
    bs_l.add(1, pcvl.BS(theta=pcvl.P("theta_l2")))
    bs_l.add(1, pcvl.PS(phi=pcvl.P("theta_l3")))
    bs_l.add(0, pcvl.BS(theta=pcvl.P("theta_l4")))
    bs_l.add(2, pcvl.PS(phi=pcvl.P("theta_l5")))

    c_var = pcvl.Circuit(m)
    for i in range(input_size):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - input_size) // 2, pcvl.PS(px))

    bs_r = pcvl.Circuit(m)
    bs_r.add(0, pcvl.BS(theta=pcvl.P("theta_r0")))
    bs_r.add(0, pcvl.PS(phi=pcvl.P("theta_r1")))
    bs_r.add(1, pcvl.BS(theta=pcvl.P("theta_r2")))
    bs_r.add(1, pcvl.PS(phi=pcvl.P("theta_r3")))
    bs_r.add(0, pcvl.BS(theta=pcvl.P("theta_r4")))
    bs_r.add(2, pcvl.PS(phi=pcvl.P("theta_r5")))

    c = pcvl.Circuit(m)
    c.add(0, bs_l, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, bs_r, merge=True)

    return c


def create_vqc_bs_mesh(m, input_size):
    """
    Create beam splitter mesh based variational quantum classifier.

    Args:
        m (int): Number of modes in the photonic circuit
        input_size (int): Number of input features to encode

    Returns:
        pcvl.Circuit: The constructed beam splitter mesh quantum circuit

    Note:
        Implementation based on https://perceval.quandela.net/docs/v0.13/notebooks/BS-based_implementation.html
    """

    bs_l = pcvl.GenericInterferometer(
        m,
        lambda idx: (
            pcvl.BS(theta=pcvl.P(f"theta_l{idx}"))
            // (0, pcvl.PS(phi=np.pi * 2 * random.random()))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=2 * m,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
    )

    c_var = pcvl.Circuit(m)
    for i in range(input_size):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - input_size) // 2, pcvl.PS(px))

    bs_r = pcvl.GenericInterferometer(
        m,
        lambda idx: (
            pcvl.BS(theta=pcvl.P(f"theta_r{idx}"))
            // (0, pcvl.PS(phi=np.pi * 2 * random.random()))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=2 * m,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
    )

    c = pcvl.Circuit(m)
    c.add(0, bs_l, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, bs_r, merge=True)

    return c


def create_vqc_general(m, input_size):
    """
    Create general interferometer based variational quantum classifier.

    Args:
        m (int): Number of modes in the photonic circuit
        input_size (int): Number of input features to encode

    Returns:
        pcvl.Circuit: The constructed general interferometer quantum circuit

    Note:
        Based on https://gitlab.quandela.dev/applications/merlin_notebooks/-/blob/main/classification/IRIS/merlin_classifier.ipynb
    """

    wl = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_li{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_lo{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c_var = pcvl.Circuit(m)
    for i in range(input_size):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - input_size) // 2, pcvl.PS(px))

    wr = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ri{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ro{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, wr, merge=True)

    return c


def get_vqc(
    m,
    input_size,
    initial_state,
    no_bunching=False,
    activation="none",
    circuit="bs_mesh",
    scale_type="learned",
):
    """
    Create a complete variational quantum classifier with specified configuration.

    Args:
        m (int): Number of modes in the photonic circuit
        input_size (int): Number of input features to encode
        initial_state (list): Initial Fock state (e.g., [1, 0, 0])
        no_bunching (bool): Whether to disable photon bunching
        activation (str): Activation function ("none", "sigmoid", "softmax")
        circuit (str): Circuit type ("bs_mesh", "general", "bs_basic", "spiral")
        scale_type (str): Input scaling method

    Returns:
        nn.Sequential: Complete VQC model ready for training

    Raises:
        ValueError: If unknown circuit type or activation function is specified
    """

    if circuit == "bs_mesh":
        vqc_circuit = create_vqc_bs_mesh(m, input_size)
    elif circuit == "general":
        vqc_circuit = create_vqc_general(m, input_size)
    elif circuit == "bs_basic":
        vqc_circuit = create_vqc_bs_basic(m, input_size)
    elif circuit == "spiral":
        vqc_circuit = create_vqc_spiral(m, input_size)
    else:
        raise ValueError(f"Unknown circuit {circuit}")

    input_layer = ScaleLayer(input_size, scale_type=scale_type)

    n_photons = torch.sum(torch.tensor(initial_state))
    if no_bunching:
        # No bunching: C(n_modes, n_photons)
        output_size = comb(m, n_photons)
    else:
        # With bunching: C(n_modes + n_photons - 1, n_photons)
        output_size = comb(m + n_photons - 1, n_photons)

    print(f"Output size of quantum layer: {output_size}")
    computation_space = (
        ComputationSpace.UNBUNCHED if no_bunching else ComputationSpace.FOCK
    )

    vqc = QuantumLayer(
        input_size=input_size,
        circuit=vqc_circuit,
        trainable_parameters=[
            p.name for p in vqc_circuit.get_parameters() if not p.name.startswith("px")
        ],
        input_parameters=["px"],
        input_state=initial_state,  # [1, 0] * 3 for example
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    )

    # The Linear layer acts as the observable and it makes sure the output is 1 dimensional

    if activation == "none":
        classification_layer = nn.Linear(vqc.output_size, 1)
        # nn.init.xavier_uniform_(classification_layer.weight)
        complete_vqc = nn.Sequential(input_layer, vqc, classification_layer)
    elif activation == "sigmoid":
        classification_layer = nn.Linear(vqc.output_size, 1)
        # nn.init.xavier_uniform_(classification_layer.weight)
        complete_vqc = nn.Sequential(
            input_layer, vqc, classification_layer, nn.Sigmoid()
        )
    elif activation == "softmax":
        classification_layer = nn.Linear(vqc.output_size, 2)
        # nn.init.xavier_uniform_(classification_layer.weight)
        complete_vqc = nn.Sequential(
            input_layer, vqc, classification_layer, nn.Softmax(dim=1)
        )
    else:
        raise ValueError(
            f"Activation function unknown or not implemented: '{activation}'"
        )

    return complete_vqc


def count_parameters(model):
    """
    Count trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to analyze

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ScaleLayer(nn.Module):
    """
    Multiply the input tensor by a learned or fixed factor.

    Args:
        dim (int): Dimension of the input data to be encoded
        scale_type (str): Type of scaling method. Options:
            - "learned": Learnable parameter initialized randomly
            - "2pi", "pi", "1": Fixed scaling factors
            - "/2pi", "/pi": Inverse scaling factors
            - "0.1", "0.5": Decimal scaling factors

    Returns:
        nn.Module that multiplies the input tensor by a scaling factor
    """

    def __init__(self, dim, scale_type="learned"):
        super().__init__()
        # Create a single learnable parameter (initialized to 1.0 by default)
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)
        elif scale_type == "/2pi":
            self.scale = torch.full((dim,), 1 / (2 * torch.pi))
        elif scale_type == "/pi":
            self.scale = torch.full((dim,), 1 / torch.pi)
        elif scale_type == "0.1":
            self.scale = torch.full((dim,), 0.1)
        elif scale_type == "0.5":
            self.scale = torch.full((dim,), 0.5)

    def forward(self, x):
        """Apply scaling to input tensor."""
        return x * self.scale
