# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.special import binom
from torch import nn

from photonic_QCNN.src.paper_layers.HW_preserving_QCNN.toolbox import (
    RBS_generalized,
    RBS_generalized_I2_2D,
    RBS_generalized_I2_3D_bottom_channel,
    map_RBS,
)


#################################################################################
### RBS decomposition toolbox:                                                  #
#################################################################################
def RBS_Unitary(nbr_state, gate_impact, device):
    """Return an RBS corresponding unitary decomposed as coeffs that should be multiplied by
    cos(theta), coeffs that should be multiplied by sin(theta) and the ones that are constant
    equal to one. This decomposition allows to avoid inplace operations.
    Args:
        - nbr_state: size of the considered basis
        - gate impact: list of tuples of basis vectors. Their planar rotation satisfies
        this transformation
        - device: torch device (cpu, cuda, etc...)
    """
    cos_matrix = torch.zeros((nbr_state, nbr_state), dtype=torch.float32, device=device)
    sin_matrix = torch.zeros((nbr_state, nbr_state), dtype=torch.float32, device=device)
    id_matrix = torch.eye(nbr_state, dtype=torch.uint8, device=device)
    for tuple_states in gate_impact:
        i, j = tuple_states
        id_matrix[i, i] = 0
        id_matrix[j, j] = 0
        cos_matrix[i, i] = 1
        cos_matrix[j, j] = 1
        sin_matrix[i, j] = 1
        sin_matrix[j, i] = -1
    return (cos_matrix, sin_matrix, id_matrix)


def Chosen_RBS_VQC(n, k, list_gates):
    """Design the particle preserving quantum circuit.
    Args:
        - n: nbr of qubits
        - k: chosen Hamming Weight
        - list_gates: list of tuples that describe the qubits on which are applied the RBS
    Outputs:
        - QNN_dictionary: dictionary that link each gate with a list of tuples that
        represents the planar rotations form this RBS
        - QNN_layer: list of list of gates. Each sub-list represents a layer filled with
        RBS gates represented by indexes.
    """
    QNN_layer, QNN_dictionary = [[i] for i in range(len(list_gates))], {}
    mapping_RBS = map_RBS(n, k)
    for RBS in range(len(QNN_layer)):
        i, j = list_gates[RBS]
        QNN_dictionary[RBS] = RBS_generalized(i, j, n, k, mapping_RBS)
    return (QNN_dictionary, QNN_layer)


def RBS_Unitaries(n, k, list_gates, device):
    """We store the RBS unitaries corresponding to each edge in the qubit connectivity to
    save memory. This allows to different RBS applied on the same pair of qubit to use the
    same unitary (but different parameters).
    Args:
        - n: nbr of qubits
        - k: chosen Hamming Weight
        - list_gates: list of tuples representing the qubits affected by each RBS
        - device: torch device (cpu, cuda, etc...)
    Output:
        - RBS_Unitaries_dict: a dictionary with key tuples of qubits affected by RBS and
        with values tuples of tensors that decompose the equivalent unitary such as in
        RBS_Unitary (cos_matrix, sin_matrix, id_matrix)
    """
    RBS_Unitaries_dict, qubit_edges = {}, list(set(list_gates))
    mapping_RBS = map_RBS(n, k)
    for i, j in qubit_edges:
        RBS_Unitaries_dict[(i, j)] = RBS_Unitary(
            int(binom(n, k)), RBS_generalized(i, j, n, k, mapping_RBS), device
        )
    return RBS_Unitaries_dict


def RBS_Unitaries_I2(I, list_gates, device):
    """We store the RBS unitaries corresponding to each edge in the qubit connectivity to
    save memory. This allows to different RBS applied on the same pair of qubit to use the
    same unitary (but different parameters). This function differs from RBS_Unitaries as
    we consider the basis of the Image.
    Args:
        - I: size of the image
        - list_gates: list of tuples representing the qubits affected by each RBS
        - device: torch device (cpu, cuda, etc...)
    Output:
        - RBS_Unitaries_dict: a dictionary with key tuples of qubits affected by RBS and
        with values tuples of tensors that decompose the equivalen unitary such as in
        RBS_Unitary (cos_matrix, sin_matrix, id_matrix)
    """
    RBS_Unitaries_dict, qubit_edges = {}, list(set(list_gates))
    for i, j in qubit_edges:
        RBS_Unitaries_dict[(i, j)] = RBS_Unitary(
            int(I**2), RBS_generalized_I2_2D(i, j, I), device
        )
    return RBS_Unitaries_dict


def RBS_Unitaries_I2_3D(I, J, list_gates, device):
    """We store the RBS unitaries corresponding to each edge in the qubit connectivity to
    save memory. This allows to different RBS applied on the same pair of qubit to use the
    same unitary (but different parameters). This function differs from RBS_Unitaries as
    we consider the basis of the Image.
    Args:
        - I: size of the image
        - list_gates: list of tuples representing the qubits affected by each RBS
        - device: torch device (cpu, cuda, etc...)
    Output:
        - RBS_Unitaries_dict: a dictionary with key tuples of qubits affected by RBS and
        with values tuples of tensors that decompose the equivalen unitary such as in
        RBS_Unitary (cos_matrix, sin_matrix, id_matrix)
    """
    RBS_Unitaries_dict, qubit_edges = {}, list(set(list_gates))
    for i, j in qubit_edges:
        RBS_Unitaries_dict[(i, j)] = RBS_Unitary(
            int(I * I * J), RBS_generalized_I2_3D_bottom_channel(i, j, I, J), device
        )
    return RBS_Unitaries_dict


#################################################################################
### RBS gate class:                                                             #
#################################################################################
class RBS_Gate_state_vector(nn.Module):
    """This module describe the action of one RBS gate."""

    def __init__(self, qubit_tuple, device):
        """Args:
        - qubit_tuple: tuple of the 2 qubits index affected by the RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)
        self.qubit_tuple = qubit_tuple

    def forward(self, input, RBS_unitaries):
        """Application of the RBS corresponding unitary on the input state.
        Args:
            - input: a torch vector representing the initial input state
            - RBS_unitaries: a dictionary that gives the RBS unitary corresponding
            to the qubit tuple such defined in RBS_Unitaries function
        Output:
            - output state from the application of the RBS on the input state
        """
        return (
            RBS_unitaries[self.qubit_tuple][0] * torch.cos(self.angle)
            + RBS_unitaries[self.qubit_tuple][1] * torch.sin(self.angle)
            + RBS_unitaries[self.qubit_tuple][2]
        ).matmul(input)


class RBS_Gate_density(nn.Module):
    """This module describe the action of one RBS gate with density matrix as input."""

    def __init__(self, qubit_tuple, device):
        """Args:
        - qubit_tuple: tuple of the 2 qubits index affected by the RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)
        self.qubit_tuple = qubit_tuple

    def forward(self, input, RBS_unitaries):
        """Application of the RBS corresponding unitary on the input state.
        Args:
            - input: a torch matrix representing the initial input as a density matrix
            - RBS_unitaries: a dictionary that gives the RBS unitary corresponding
            to the qubit tuple such defined in RBS_Unitaries function
        Output:
            - output density matrix from the application of the RBS on the input state
        """
        return (
            (
                RBS_unitaries[self.qubit_tuple][0] * torch.cos(self.angle)
                + RBS_unitaries[self.qubit_tuple][1] * torch.sin(self.angle)
                + RBS_unitaries[self.qubit_tuple][2]
            )
            .matmul(input)
            .matmul(
                (
                    RBS_unitaries[self.qubit_tuple][0] * torch.cos(self.angle)
                    + RBS_unitaries[self.qubit_tuple][1] * torch.sin(self.angle)
                    + RBS_unitaries[self.qubit_tuple][2]
                ).t()
            )
        )


#################################################################################
### RBS based VQC class:                                                        #
#################################################################################
class RBS_VQC_state_vector(nn.Module):
    """This module describes the action of one RBS based VQC."""

    def __init__(self, n, k, list_gates, device):
        """Args:
        - n: nbr of qubits
        - k: chosen Hamming Weight
        - list_gates: list of tuples representing the qubits affected by each RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries(n, k, list_gates, device)
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Gate_state_vector(list_gates[i], device)
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based VQC.
        Arg:
            - input_state = a state vector on which is applied the RBS from the VQC
        Output:
            - final state from the application of the RBS from the VQC on the input
            state
        """
        input_state = input_state.unsqueeze(-1)
        for RBS in self.RBS_gates:
            input_state = RBS(input_state, self.RBS_Unitaries_dict)
        input_state = input_state.squeeze(-1)
        return input_state


class RBS_VQC_density(nn.Module):
    """This module describes the action of one RBS based VQC with density matrix
    as input."""

    def __init__(self, n, k, list_gates, device):
        """Args:
        - n: nbr of qubits
        - k: chosen Hamming Weight
        - list_gates: list of tuples representing the qubits affected by each RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries(n, k, list_gates, device)
        self.RBS_gates = nn.ModuleList(
            [RBS_Gate_density(list_gates[i], device) for i in range(len(list_gates))]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based VQC.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for RBS in self.RBS_gates:
            input_state = RBS(input_state, self.RBS_Unitaries_dict)
        return input_state
