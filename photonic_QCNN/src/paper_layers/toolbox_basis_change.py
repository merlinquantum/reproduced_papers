# ruff: noqa: N801, N806, N802
import torch
import torch.nn as nn
from qoptcraft.basis import get_photon_basis, hilbert_dim


def Basis_dictionary_Image_to_Fock(d1, d2):
    """This function returns the dictionary that links the Fock basis to the Image basis.
    We consider a 2-dimensional image on two registers of size d1 and d2."""
    dictionary = {}
    Fock_basis = get_photon_basis(d1 + d2, 2)
    for i in range(d1):
        for j in range(d2):
            Fock_state = [0 for i in range(d1 + d2)]
            Fock_state[i] = 1
            Fock_state[d1 + j] = 1
            dictionary[i * d2 + j] = Fock_basis.index(tuple(Fock_state))
    return dictionary


def Basis_dictionary_Image_to_larger_Fock(d1, d2, add1, add2):
    """This function returns the dictionary that links the Fock basis to the Image basis.
    We consider a 2-dimensional image on two registers of size d1 and d2."""
    dictionary = {}
    Fock_basis = get_photon_basis(d1 + d2 + add1 + add2, 2)
    for i in range(d1):
        for j in range(d2):
            Fock_state = [0 for i in range(d1 + d2 + add1 + add2)]
            Fock_state[add1 + i] = 1
            Fock_state[add1 + d1 + j] = 1
            dictionary[i * d2 + j] = Fock_basis.index(tuple(Fock_state))
    return dictionary


def Projector_Fock_to_Image(d1, d2, device):
    """This function returns the projector from the Fock basis to the Image basis.
    Args:
        - d1: the image must be of size d1 x d2
        - d2: the image must be of size d1 x d2
        - device: torch device (cpu, cuda, etc...)
    Output:
        - P a projector that maps a state in the Fock basis into the image basis
    """
    d1 + d2  # number of modes


def Passage_matrix_Image_larger_Fock(d1, d2, add1, add2, device):
    """This function returns the passage matrix that links the Image basis to the Fock basis.
    We consider a 2-dimensional image on two registers of size d1 and d2."""
    dimension = hilbert_dim(d1 + d2 + add1 + add2, 2)
    Passage_matrix = torch.zeros((dimension, d1 * d2), dtype=torch.uint8, device=device)
    Fock_Image_dict = Basis_dictionary_Image_to_larger_Fock(d1, d2, add1, add2)
    for i in range(d1):
        for j in range(d2):
            Passage_matrix[Fock_Image_dict[i * d2 + j], i * d2 + j] = 1
    return Passage_matrix


def Passage_matrix_Image_Fock(d1, d2, device):
    """This function returns the passage matrix that links the Image basis to the Fock basis.
    We consider a 2-dimensional image on two registers of size d1 and d2."""
    dimension = hilbert_dim(d1 + d2, 2)
    Passage_matrix = torch.zeros((dimension, d1 * d2), dtype=torch.uint8, device=device)
    Fock_Image_dict = Basis_dictionary_Image_to_Fock(d1, d2)
    for i in range(d1):
        for j in range(d2):
            Passage_matrix[Fock_Image_dict[i * d2 + j], i * d2 + j] = 1
    return Passage_matrix


class Basis_Change_Image_to_Fock_state_vector(nn.Module):
    """This module allows to change the basis from the Image basis to the HW basis."""

    def __init__(self, d1, d2, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.Passage_matrix = Passage_matrix_Image_Fock(d1, d2, device)

    def forward(self, input_state):
        """This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state..
        Output:
            - a torch density operator that represents the output mixted state in
            the Fock basis (with 2 photons).
        """
        return self.Passage_matrix.to(torch.float32) @ input_state


class Basis_Change_Image_to_Fock_density(nn.Module):
    """This module allows to change the basis from the Image basis to the HW basis."""

    def __init__(self, d1, d2, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.Passage_matrix = Passage_matrix_Image_Fock(d1, d2, device)

    def forward(self, input_state):
        """This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state..
        Output:
            - a torch density operator that represents the output mixted state in
            the Fock basis (with 2 photons).
        """
        return (
            self.Passage_matrix.to(torch.float32)
            @ input_state
            @ self.Passage_matrix.to(torch.float32).T
        )


class Basis_Change_Image_to_larger_Fock_density(nn.Module):
    """This module allows to change the basis from the Image basis to the HW basis."""

    def __init__(self, d1, d2, add1, add2, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.Passage_matrix = Passage_matrix_Image_larger_Fock(
            d1, d2, add1, add2, device
        )

    def forward(self, input_state):
        """This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state..
        Output:
            - a torch density operator that represents the output mixted state in
            the Fock basis (with 2 photons).
        """
        return (
            self.Passage_matrix.to(torch.float32)
            @ input_state
            @ self.Passage_matrix.to(torch.float32).T
        )


class Basis_Change_Image_to_larger_Fock_state_vector(nn.Module):
    """This module allows to change the basis from the Image basis to the HW basis."""

    def __init__(self, d1, d2, add1, add2, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.Passage_matrix = Passage_matrix_Image_larger_Fock(
            d1, d2, add1, add2, device
        )

    def forward(self, input_state):
        """This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state..
        Output:
            - a torch density operator that represents the output mixted state in
            the Fock basis (with 2 photons).
        """
        return self.Passage_matrix.to(torch.float32) @ input_state
