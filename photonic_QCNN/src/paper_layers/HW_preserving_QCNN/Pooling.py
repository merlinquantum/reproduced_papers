# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
import torch
from torch import nn


def Pooling_2D_Projector_HW(I, O, device):
    """This function mimics the effect of a HW preserving pooling layer on half
    the remaining qubits. We suppose that the input image is square.
    Args:
        - I: size of the input image
        - O: size of the output image (we suppose O = I//2 for now)
        - device: torch device (cpu, cuda, etc...)
    Output:
        - Projector: projector corresponding to all cases of measurement. Its
        dimension is (k, O**2, I**2) with k the number of cases of measurement.
    """
    # Number of matrices:
    Number_of_matrices, index = 1 + 2 * O + O**2, 0
    # We consider the case where all measured qubits are on state |0>:
    Projectors = torch.zeros(
        (Number_of_matrices, O**2, I**2), dtype=torch.uint8, device=device
    )
    for i in range(O):
        for j in range(O):
            Projectors[index, i * O + j, (i * 2 + 1) * I + (j * 2 + 1)] = 1
    index += 1
    # We consider the case where we measured 2 qubits in state |1>:
    for i in range(O):
        for j in range(O):
            Projectors[index, i * O + j, (i * 2) * I + (j * 2)] = 1
            index += 1
    # We finally consider the case where we measured only one qubit in state |1>:
    # If we measure a qubit in |1> in the line register
    for i in range(O):  # we measure this qubit in state |1>
        for j in range(O):
            Projectors[index, i * O + j, (2 * i) * I + 2 * j + 1] = 1
        index += 1
    # If we measure a qubit in |1> in the column register
    for j in range(O):  # we measure this qubit in state |1>
        for i in range(O):
            Projectors[index, i * O + j, (2 * i + 1) * I + 2 * j] = 1
        index += 1
    return Projectors


def Pooling_2D_Projector_HW_reverse(I, O, device):
    """This function mimics the effect of a HW preserving pooling layer on half
    the remaining qubits. We suppose that the input image is square. The difference
    with the previous projector Pooling_2D_Projector_HW is that the pooling on the
    second register is reversed.
    Args:
        - I: size of the input image
        - O: size of the output image (we suppose O = I//2 for now)
        - device: torch device (cpu, cuda, etc...)
    Output:
        - Projector: projector corresponding to all cases of measurement. Its
        dimension is (k, O**2, I**2) with k the number of cases of measurement.
    """
    # Number of matrices:
    Number_of_matrices, index = 1 + 2 * O + O**2, 0
    # We consider the case where all measured qubits are on state |0>:
    Projectors = torch.zeros(
        (Number_of_matrices, O**2, I**2), dtype=torch.uint8, device=device
    )
    for i in range(O):
        for j in range(O):
            Projectors[index, i * O + j, (i * 2 + 1) * I + (j * 2 + 1)] = 1
    index += 1
    # We consider the case where we measured 2 qubits in state |1>:
    for i in range(O):
        for j in range(O):
            Projectors[index, i * O + j, (i * 2) * I + (j * 2)] = 1
            index += 1
    # We finally consider the case where we measured only one qubit in state |1>:
    # If we measure a qubit in |1> in the line register
    for i in range(O):  # we measure this qubit in state |1>
        for j in range(O):
            Projectors[index, i * O + j, (2 * i) * I + 2 * j + 1] = 1
        index += 1
    # If we measure a qubit in |1> in the column register
    for j in range(O):  # we measure this qubit in state |1>
        for i in range(O):
            Projectors[index, i * O + j, (2 * i + 1) * I + 2 * j] = 1
        index += 1
    return Projectors


def Pooling_3D_Projector_HW(I, O, J, device):
    """This function mimics the effect of a HW preserving pooling layer on half
    the remaining qubits. We suppose that the input image is square.
    Args:
        - I: size of the input image
        - O: size of the output image (we suppose O = I//2 for now)
        - device: torch device (cpu, cuda, etc...)
    Output:
        - Projector: projector corresponding to all cases of measurement. Its
        dimension is (k, O**2, I**2) with k the number of cases of measurement.
    """
    # Number of matrices:
    Number_of_matrices, index = 1 + 2 * O + O**2, 0
    # We consider the case where all measured qubits are on state |0>: O=2, J=4
    Projectors = torch.zeros(
        (Number_of_matrices, O * O * J, I * I * J), dtype=torch.uint8, device=device
    )  # (16,64)
    for i in range(O):
        for j in range(O):
            for c in range(J):
                Projectors[
                    index, (i * O + j) * J + c, ((i * 2 + 1) * I + (j * 2 + 1)) * J + c
                ] = 1
    index += 1
    # We consider the case where we measured 2 qubits in state |1>:
    for i in range(O):
        for j in range(O):
            for c in range(J):
                Projectors[
                    index, (i * O + j) * J + c, ((i * 2) * I + (j * 2)) * J + c
                ] = 1
            index += 1
    # We finally consider the case where we measured only one qubit in state |1>:
    # If we measure a qubit in |1> in the line register
    for i in range(O):  # we measure this qubit in state |1>
        for j in range(O):
            for c in range(J):
                Projectors[
                    index, (i * O + j) * J + c, ((2 * i) * I + 2 * j + 1) * J + c
                ] = 1
        index += 1
    # If we measure a qubit in |1> in the column register
    for j in range(O):  # we measure this qubit in state |1>
        for i in range(O):
            for c in range(J):
                Projectors[
                    index, (i * O + j) * J + c, ((2 * i + 1) * I + 2 * j) * J + c
                ] = 1
        index += 1
    return Projectors


class Pooling_2D_state_vector(nn.Module):
    """This module describe the effect of the Pooling on the QCNN architecture."""

    def __init__(self, I, O, device):
        """We suppose that the input image is square."""
        super().__init__()
        self.Projectors = Pooling_2D_Projector_HW(I, O, device)
        self.Number_of_states = (
            1 + 2 * O + O**2
        )  # Number of projectors in Projectors_matrix
        self.O = O

    def forward(self, input_state):
        """This module forward a tensor made of each pure state weighted by their
        probabilities that describe the output mixed state form the pooling layer.
        Arg:
            - input_state: a torch vector representing the initial input state. Its
            dimension is (nbr_batch, I**2).
        Output:
            - a torch vector made of several vectors that represents the output
            mixted state with dimension (nbr_batch*k, O**2) with k the number of
            pure states representing the mixed state.
        """
        input_state = torch.einsum(
            "bi, koi->bko", input_state, self.Projectors.to(torch.float32)
        )
        # Resize the new state from dimension (nbr_batch, k, O**2) to dimension (nbr_batch*k, O**2):
        input_state = input_state.view(-1, self.O**2)
        return input_state


class Pooling_2D_density_HW(nn.Module):
    """This module describe the effect of the Pooling on the QCNN architecture while
    simulating states as density operators."""

    def __init__(self, I, O, device):
        """We suppose that the input image is square."""
        super().__init__()
        self.Projectors = Pooling_2D_Projector_HW(I, O, device).float()
        self.O = O
        self.device = device

    def forward(self, input):
        """This module forward a tensor made of each pure state weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state (density matrix).
            Its dimension is (nbr_batch, I**2, I**2).
        Output:
            - a torch vector density operator that represents the output mixted
            state with dimension (nbr_batch, O**2, O**2).
        """
        mixed_state_density_matrix = torch.zeros(
            input.size()[0], self.O**2, self.O**2
        ).to(self.device)
        for i in range(input.size()[0]):
            for p in self.Projectors:
                mixed_state_density_matrix[i] += p @ input[i] @ p.T
        input = mixed_state_density_matrix
        return input


class Pooling_3D_density(nn.Module):
    """This module describe the effect of the Pooling on the QCNN architecture while
    simulating states as density operators."""

    def __init__(self, I, O, J, device):
        """We suppose that the input image is square."""
        super().__init__()
        self.Projectors = Pooling_3D_Projector_HW(I, O, J, device).float()
        self.O = O
        self.J = J
        self.device = device

    def forward(self, input):
        """This module forward a tensor made of each pure state weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state (density matrix).
            Its dimension is (nbr_batch, I**2, I**2).
        Output:
            - a torch vector density operator that represents the output mixted
            state with dimension (nbr_batch, O**2, O**2).
        """
        mixed_state_density_matrix = torch.zeros(
            input.size()[0], (self.O**2) * self.J, (self.O**2) * self.J
        ).to(self.device)
        for i in range(input.size()[0]):
            for p in self.Projectors:
                mixed_state_density_matrix[i] += p @ input[i] @ p.T
        input = mixed_state_density_matrix
        return input


def Pooling_3D_Projector_channel_HW(I, O, J, L, device):
    """This function mimics the effect of a HW preserving 3D channel pooling layer on half
    the remaining qubits. We suppose that the input image is square.
    Args:
        - I: size of the input image
        - O: size of the output image (we suppose O = I//2 for now)
        - device: torch device (cpu, cuda, etc...)
    Output:
        - Projector: projector corresponding to all cases of measurement. Its
        dimension is (k, O**2, I**2) with k the number of cases of measurement.
    """
    # Number of matrices:
    Number_of_matrices, index = (1 + 2 * O + O**2) * J, 0
    # We consider the case where all measured qubits are on state |0>: O=2, J=4
    Projectors = torch.zeros(
        (Number_of_matrices, O * O * L, I * I * J), dtype=torch.uint8, device=device
    )  # (16,64)
    for c in range(J):
        for i in range(O):
            for j in range(O):
                Projectors[
                    index,
                    ((i * O + j) * J + c) // 2,
                    ((i * 2 + 1) * I + (j * 2 + 1)) * J + c,
                ] = 1
        index += 1
    # We consider the case where we measured 2 qubits in state |1>:
    for i in range(O):
        for j in range(O):
            for c in range(J):
                Projectors[
                    index, ((i * O + j) * J + c) // 2, ((i * 2) * I + (j * 2)) * J + c
                ] = 1
                index += 1
    # We finally consider the case where we measured only one qubit in state |1>:
    # If we measure a qubit in |1> in the line register
    for i in range(O):
        for c in range(J):
            for j in range(O):
                Projectors[
                    index, ((i * O + j) * J + c) // 2, ((2 * i) * I + 2 * j + 1) * J + c
                ] = 1
            index += 1
    # If we measure a qubit in |1> in the column register
    for j in range(O):  # we measure this qubit in state |1>
        for c in range(J):
            for i in range(O):
                Projectors[
                    index, ((i * O + j) * J + c) // 2, ((2 * i + 1) * I + 2 * j) * J + c
                ] = 1
            index += 1
    return Projectors


class Pooling_3D_density_channel_HW(nn.Module):
    """This module describe the effect of the Pooling on the QCNN architecture while
    simulating states as density operators."""

    def __init__(self, I, O, J, device):
        """We suppose that the input image is square."""
        super().__init__()
        self.Projectors = Pooling_3D_Projector_channel_HW(
            I, O, J, J // 2, device
        ).float()
        self.O = O
        self.J = J // 2
        self.device = device

    def forward(self, input):
        """This module forward a tensor made of each pure state weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state (density matrix).
            Its dimension is (nbr_batch, I**2, I**2).
        Output:
            - a torch vector density operator that represents the output mixted
            state with dimension (nbr_batch, O**2, O**2).
        """
        mixed_state_density_matrix = torch.zeros(
            input.size()[0], (self.O**2) * self.J, (self.O**2) * self.J
        ).to(self.device)
        for i in range(input.size()[0]):
            for p in self.Projectors:
                mixed_state_density_matrix[i] += p @ input[i] @ p.T
        input = mixed_state_density_matrix
        return input
