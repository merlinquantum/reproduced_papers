# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
import torch
from torch import nn


def Linear_Pooling_2D_Projector_HW(I, O, device):
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
    Linear_Projector = torch.zeros((O**2, I**2), dtype=torch.uint8, device=device)
    for proj in Projectors:
        Linear_Projector += proj
    return Linear_Projector


class Linear_Pooling_2D_state_vector(nn.Module):
    """This module describe the effect of the Pooling on the QCNN architecture."""

    def __init__(self, I, O, device):
        """We suppose that the input image is square."""
        super().__init__()
        self.Projector = Linear_Pooling_2D_Projector_HW(I, O, device)
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
        return self.Projector.to(torch.float32).matmul(input_state)
