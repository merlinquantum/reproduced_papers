# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
"""
===================================================================================
This file consists of 4 parts:
    - RBS gate class for convolution: single Conv RBS gate classes
    - Convolutional layer class in the basis of fixed HW: Conv layer for HW basis, pyramid kernel layout
    - 2D Convolutional layer class in the Image basis: Conv layer for 2D Image basis, pyramid kernel layout
    - 3D Convolutional layer class in the Image basis with customized kernel layout: Conv layer for 3D Image basis with customized kernel layout
===================================================================================
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from photonic_QCNN.src.paper_layers.HW_preserving_QCNN.RBS_Circuit import (
    RBS_Unitaries,
    RBS_Unitaries_I2,
    RBS_Unitaries_I2_3D,
)
from photonic_QCNN.src.paper_layers.HW_preserving_QCNN.toolbox import (
    QCNN_RBS_based_VQC,
    QCNN_RBS_based_VQC_3D,
)


#################################################################################
### RBS gate class for convolution:                                             #
#################################################################################
class RBS_Conv_state_vector(nn.Module):
    """This module describe the action of one RBS gate in the Conv Layer.
    The only change with the RBS_Gate_state_vector module is that you choose the RBS
    corresponding parameter."""

    def __init__(self, qubit_tuple, theta, device):
        """Args:
        - qubit_tuple: tuple of the 2 qubits index affected by the RBS
        - theta: torch parameter of the RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.angle = theta
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


class RBS_Conv_density(nn.Module):
    """This module describe the action of one RBS gate in the Conv Layer.
    The only change with the RBS_Gate_density module is that you choose the RBS
    corresponding parameter."""

    def __init__(self, qubit_tuple, theta, device):
        """Args:
        - qubit_tuple: tuple of the 2 qubits index affected by the RBS
        - theta: torch parameter of the RBS
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.angle = theta
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
### Convolutional layer class in the basis of fixed HW:                         #
#################################################################################
class Conv_RBS_state_vector(nn.Module):
    """This module describes the action of an RBS based convolutional layer."""

    def __init__(self, I, K, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(I, K)
        for key in RBS_dictionary:
            list_gates.append((RBS_dictionary[key], RBS_dictionary[key] + 1))
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries(I * 2, 2, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for i in range(int(K * (K - 1)))
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_state_vector(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
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


class Conv_RBS_density(nn.Module):
    """This module describes the action of a RBS based convolutional layer."""

    def __init__(self, I, K, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(I, K)
        for key in RBS_dictionary:
            list_gates.append((RBS_dictionary[key], RBS_dictionary[key] + 1))
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries(I * 2, 2, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for i in range(int(K * (K - 1)))
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_density(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
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


#################################################################################
### 2D Convolutional layer class in the Image basis :                              #
#################################################################################
class Conv_RBS_state_vector_I2(nn.Module):
    """This module describes the action of a RBS based convolutional layer in the basis of the Image."""

    def __init__(self, I, K, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(I, K)
        for key in RBS_dictionary:
            list_gates.append(RBS_dictionary[key])
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries_I2(I, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for i in range(int(K * (K - 1)))
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_state_vector(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
        Arg:
            - input_state = an initial state vector on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for RBS in self.RBS_gates:
            input_state = RBS(input_state, self.RBS_Unitaries_dict)
        return input_state


class Conv_RBS_density_I2_Parameters(nn.Module):
    """This module describes the action of a RBS based convolutional layer in the basis of the Image."""

    def __init__(self, I, K, Conv_Params, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(I, K)
        for key in RBS_dictionary:
            list_gates.append(RBS_dictionary[key])
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries_I2(I, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(Conv_Params[i], requires_grad=False)
                for i in range(len(Conv_Params))
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_density(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
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


class Conv_RBS_density_I2(nn.Module):
    """This module describes the action of a RBS based convolutional layer in the basis of the Image."""

    def __init__(self, I, K, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(I, K)
        for key in RBS_dictionary:
            list_gates.append(RBS_dictionary[key])
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries_I2(I, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for i in range(int(K * (K - 1)))
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_density(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
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


########################################################################################
### 3D Convolutional layer class in the Image basis with customized kernel layout     #
########################################################################################


class Conv_RBS_density_I2_3D(nn.Module):
    """This module describes the action of a RBS based convolutional layer in the basis of the Image."""

    def __init__(self, I, K, J, kernel_layout, device):
        """Args:
        - I: dimension of the initial image (for a square image I=n/2 the number of qubits)
        - K: size of the convolutional filter
        - kernel_layout: layout of convolution kernel, could be "pyramid", "all_connection", etc. You can also check list_gates.py for details.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC_3D(
            I, K, J, kernel_layout
        )
        for key in RBS_dictionary:
            list_gates.append(RBS_dictionary[key])
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity:
        self.RBS_Unitaries_dict = RBS_Unitaries_I2_3D(I, J, list_gates, device)
        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for i in range(int(K * (K - 1)) + J * (J - 1) // 2)
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                RBS_Conv_density(
                    list_gates[i], self.Parameters[Param_dictionary[i]], device
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        """Feedforward of the RBS based Convolutional layer.
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
