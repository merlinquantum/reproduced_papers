"""
Boson sampler utilities for the DQNN photonic quantum train.

This module defines a boson sampler wrapper that builds parameterized
photonic circuits and exposes a QuantumLayer for training.
"""

from math import comb, pi

import merlin as ML
import perceval as pcvl
import torch


## Quantum-Train Inspired ##
class BosonSampler:
    """
    Boson sampler with parameterized photonic circuits for quantum training.

    Parameters
    ----------
    m : int
        Number of modes in the photonic circuit.
    n : int
        Number of photons input into the circuit.
    qnn_layers : int, optional
        Number of interferometer layers to stack. Default is 1.
    """

    def __init__(self, m: int, n: int, qnn_layers: int = 1):
        """
        Initialize the boson sampler and create its quantum layer.

        Parameters
        ----------
        m : int
            Number of modes in the photonic circuit.
        n : int
            Number of photons input into the circuit.
        qnn_layers : int, optional
            Number of interferometer layers to stack. Default is 1.
        """
        self.m = m
        self.n = n
        assert n <= m, (
            "Got more modes than photons, can only input 0 or 1 photon per mode"
        )
        self.quantum_layer = self.create_quantum_layer(qnn_layers=qnn_layers)

    @property
    def _nb_parameters_needed(self) -> int:
        """
        Number of parameters (theta, phi_i, phi_j) in the circuit.

        Returns
        -------
        int
            Total number of trainable parameters required by the circuit.
        """
        # Each beam splitter has theta, and each connected mode has a phase shifter
        # Number of beam splitters in Clements decomposition: m(m-1)/2
        # Thus, total parameters: m(m-1)/2 * 3
        return (self.m * (self.m - 1)) // 2 * 3

    @property
    def nb_parameters(self) -> int:
        """
        Maximum number of values in the input tensor.

        Returns
        -------
        int
            Total number of parameters required by the circuit.
        """
        return self._nb_parameters_needed

    @property
    def embedding_size(self) -> int:
        """
        Size of the output probability distribution.

        Returns
        -------
        int
            Number of output probabilities for the Fock basis.
        """

        return comb(self.m, self.n)

    def create_quantum_circuit(self, qnn_layers: int = 1) -> pcvl.Circuit:
        """
        Creates a parametrized interferometer using Clements decomposition.

        Parameters
        ----------
        qnn_layers : int, optional
            Number of interferometer layers to stack. Default is 1.

        Returns
        -------
        pcvl.Circuit
            Parameterized Perceval circuit.
        """
        width = len(str(self.nb_parameters - 1))
        parameters = [pcvl.P(f"phi{i:0{width}d}") for i in range(self.nb_parameters)]

        circuit = pcvl.Circuit(self.m)
        param_idx = 0

        num_layers = self.m - 1
        for qnn_layer in range(qnn_layers):
            if not qnn_layer == 0:
                for mode in range(self.m):
                    circuit.add(mode, pcvl.PS(pi))  ## non-linear layer
            for layer in range(num_layers):
                # Determine starting mode based on layer parity for checkerboard pattern
                if layer % 2 == 0:
                    mode_start = 0
                else:
                    mode_start = 1

                for mode in range(mode_start, self.m - 1, 2):
                    i = mode
                    j = mode + 1

                    # Add Beam Splitter with theta parameter
                    bs = pcvl.BS(
                        theta=parameters[param_idx],
                        phi_bl=0,
                        phi_br=0,
                        phi_tl=0,
                        phi_tr=0,
                    )
                    circuit.add((i, j), bs)
                    param_idx += 1

                    # Add Phase Shifters to modes i and j
                    circuit.add(i, pcvl.PS(parameters[param_idx]))
                    param_idx += 1

                    circuit.add(j, pcvl.PS(parameters[param_idx]))
                    param_idx += 1

        self.num_effective_params = param_idx

        return circuit

    def create_quantum_layer(self, qnn_layers: int = 1) -> ML.QuantumLayer:
        """
        Create the QuantumLayer wrapper for the parameterized circuit.

        Parameters
        ----------
        qnn_layers : int, optional
            Number of interferometer layers to stack. Default is 1.

        Returns
        -------
        merlin.QuantumLayer
            Quantum layer configured with the circuit and input state.
        """
        circuit = self.create_quantum_circuit(qnn_layers=qnn_layers)

        # Create the input state
        input_state = self.m * [0]
        places = torch.linspace(0, self.m - 1, self.n)
        for photon in places:
            input_state[int(photon)] = 1
        input_state = pcvl.BasicState(input_state)

        # Create parameters
        width = len(str(self.nb_parameters - 1))
        parameters = [f"phi{i:0{width}d}" for i in range(self.num_effective_params)]

        quantum_layer_kwargs = {
            "input_size": 0,
            "n_photons": self.n,
            "circuit": circuit,
            "input_state": input_state,
            "trainable_parameters": parameters,
        }

        # Merlin >=0.3 deprecates passing computation_space directly.
        if hasattr(ML, "MeasurementStrategy") and hasattr(
            ML.MeasurementStrategy, "probs"
        ):
            quantum_layer_kwargs["measurement_strategy"] = ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.UNBUNCHED
            )
        else:
            quantum_layer_kwargs["computation_space"] = ML.ComputationSpace.UNBUNCHED

        return ML.QuantumLayer(**quantum_layer_kwargs)

    def set_params(self, params: torch.Tensor) -> None:
        """
        Load a flat parameter vector into the quantum layer.

        Parameters
        ----------
        params : torch.Tensor
            Flat list or tensor of parameters to assign.

        Returns
        -------
        None
        """
        index = 0
        with torch.no_grad():
            for p in self.quantum_layer.parameters():
                n = p.numel()
                p.copy_(torch.tensor(list(params[index : index + n])))
                index += n
