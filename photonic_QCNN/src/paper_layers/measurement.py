# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
import torch
import torch.nn as nn
from qoptcraft.basis import get_photon_basis, hilbert_dim

"""
The measurement operation.
"""


def measurement(batch_density_matrix, device):
    """
    Args:
        - batch_density_matrix: the final density matrices with batch
        - device: torch device (cpu, cuda, etc...)
    Output:
        - The diagonal vectors of input matrices, corresponding to the sampling probability distribution
    """
    return torch.stack(
        [torch.diag(density_matrix) for density_matrix in batch_density_matrix]
    ).to(device)


def Projector_Matrix(n, m, state_indexes, device):
    """
    Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - state_indexes: list of indexes corresponding to the measured states
        - device: torch device (cpu, cuda, etc...)
    """
    dimension = hilbert_dim(m, n)
    P = torch.zeros((dimension, len(state_indexes)), dtype=torch.float32, device=device)
    for i, index in enumerate(state_indexes):
        P[index, i] = 1
    return P


def Projector_Photon_detection(n, m, modes_detected, device):
    """
    Args:
        - n (int): number of photons in the system
        - m (int): number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - device: torch device (cpu, cuda, etc...)
    """
    photon_basis = get_photon_basis(m, n)
    dimension = hilbert_dim(m, n)
    P = torch.zeros((dimension, 2), dtype=torch.float32, device=device)
    for i, state in enumerate(photon_basis):
        photon_in_mode = False
        for mode in modes_detected:
            if state[mode] > 0:
                photon_in_mode = True
                break
        if photon_in_mode:
            P[i, 0] = 1
        else:
            P[i, 1] = 1
    return P


def Projector_Photon_detection_Post_Processing(
    n, m, modes_detected, modes_detected_post_selection, values, device
):
    """
    Args:
        - n (int): number of photons in the system
        - m (int): number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - modes_detected_post_selection (list): list of modes (int) corresponding to the post-selection
        - values (tuple): tuple of number of photons in each mode detected for post-selection
        - device: torch device (cpu, cuda, etc...)
    """
    photon_basis = get_photon_basis(m, n)
    dimension = hilbert_dim(m, n)
    P = torch.zeros((dimension, 2), dtype=torch.float32, device=device)
    for i, state in enumerate(photon_basis):
        photon_in_mode = False
        photon_well_post_selected = True
        for mode in modes_detected:
            if state[mode] > 0:
                photon_in_mode = True
                break
        for j, mode in enumerate(modes_detected_post_selection):
            if state[mode] != values[j]:
                photon_well_post_selected = False
                break
        if photon_well_post_selected:
            if photon_in_mode:
                P[i, 0] = 1
            else:
                P[i, 1] = 1
    return P


def Projector_Before_Pooling(n, m, modes_detected, values, device):
    """
    Args:
        - n (int): number of photons in the system
        - m (int): number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - values (list): list of values (tuples) of number of photon in each mode detected
        - device: torch device (cpu, cuda, etc...)
    """
    photon_basis = get_photon_basis(m, n)
    dimension = hilbert_dim(m, n)
    P = torch.zeros((dimension, len(values)), dtype=torch.float32, device=device)
    for i, state in enumerate(photon_basis):
        # Checking if this state is part of the accepted values:
        for j, value in enumerate(values):
            state_detected = True
            for index, mode in enumerate(modes_detected):
                if state[mode] != value[index]:
                    state_detected = False
                    break
            if state_detected:
                P[i, j] = 1
    return P


class Measure_Projector_Fock_basis(nn.Module):
    """This class describes the action of a measurement in the Fock basis on a density matrix."""

    def __init__(self, n, m, state_indexes, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - state_indexes: list of indexes corresponding to the measured states
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.n = n
        self.m = m
        self.state_indexes = state_indexes
        self.device = device
        self.P = Projector_Matrix(n, m, state_indexes, device)

    def forward(self, input):
        """Feedforward of the measurement layer.
        Arg:
            - input = a density matrix on which is applied the measurement
        Output:
            - density matrix of the measured state
        """
        return torch.stack(
            [
                torch.matmul(self.P.T, torch.diag(density_matrix))
                for density_matrix in input
            ]
        ).to(self.device)  # Get the diagonal of the density matrix


class Measure_Photon_detection_state_vector(nn.Module):
    """This class describes the action of a measurement that detect if photons are in a mode corresponding to a label."""

    def __init__(self, n, m, modes_detected, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.n = n
        self.m = m
        self.modes_detected = modes_detected
        self.device = device
        self.P = Projector_Photon_detection(n, m, modes_detected, device)

    def forward(self, input):
        """Feedforward of the measurement layer.
        Arg:
            - input = a state vector on which is applied the measurement
        Output:
            - vector of the measured state
        """
        return torch.matmul(self.P.T, input)


class Measure_Photon_detection(nn.Module):
    """This class describes the action of a measurement that detect if photons are in a mode corresponding to a label."""

    def __init__(self, n, m, modes_detected, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.n = n
        self.m = m
        self.modes_detected = modes_detected
        self.device = device
        self.P = Projector_Photon_detection(n, m, modes_detected, device)

    def forward(self, input):
        """Feedforward of the measurement layer.
        Arg:
            - input = a density matrix on which is applied the measurement
        Output:
            - density matrix of the measured state
        """
        return torch.stack(
            [
                torch.matmul(self.P.T, torch.diag(density_matrix))
                for density_matrix in input
            ]
        ).to(self.device)  # Get the diagonal of the density matrix


class Measure_Photon_detection_Post_Selection(nn.Module):
    """This class describes the action of a measurement that detect if photons are in a mode corresponding to a label."""

    def __init__(
        self, n, m, modes_detected, modes_detected_post_selection, values, device
    ):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - modes_detected (list): list of modes (int) corresponding to the label
        - modes_detected_post_selection (list): list of modes (int) corresponding to the post-selection
        - values (tuple): tuple of number of photons in each mode detected for post-selection
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.n = n
        self.m = m
        self.modes_detected = modes_detected
        self.modes_detected_post_selection = modes_detected_post_selection
        self.values = values
        self.device = device
        if len(values) == 0:
            self.P = Projector_Photon_detection(n, m, modes_detected, device)
        else:
            self.P = Projector_Photon_detection_Post_Processing(
                n, m, modes_detected, modes_detected_post_selection, values, device
            )

    def forward(self, input):
        """Feedforward of the measurement layer.
        Arg:
            - input = a density matrix on which is applied the measurement
        Output:
            - density matrix of the measured state
        """
        return torch.stack(
            [
                torch.matmul(self.P.T, torch.diag(density_matrix))
                for density_matrix in input
            ]
        ).to(self.device)  # Get the diagonal of the density matrix


class Measure_Possible_Outcome_SI(nn.Module):
    """This custom class allows one to know each state-injection outcome when considering the photonic QCNN
    architecture. We consider tensor input of dimension 2."""

    def __init__(self, m, n, modes_detected, values, device):
        """
        Args:
            - m (int): number of modes in the system
            - n (int): number of photons in the system
            - modes_detected (list): list of modes (int) corresponding to the measured mode in the pooling
            - values (list): list of values (tuples) of the measurements during the Pooling
            - device: torch device (cpu, cuda, etc...)
        Output:
            - Probability_Outcome (dict): dictionary containing the probability of each possible outcome
        """
        super().__init__()
        self.m = m
        self.modes_detected = modes_detected
        self.device = device
        self.values = values
        self.Projector = Projector_Before_Pooling(n, m, modes_detected, values, device)

    def forward(self, input):
        """Feedforward of the measurement layer.
        Arg:
            - input = a density matrix on which is applied the measurement
        Output:
            - density matrix of the measured state
        """
        return torch.stack(
            [
                torch.matmul(self.Projector.T, torch.diag(density_matrix))
                for density_matrix in input
            ]
        ).to(self.device)
