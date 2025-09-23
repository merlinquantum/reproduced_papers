# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
from torch import nn

from photonic_QCNN.src.paper_layers.toolbox import *


def get_PVM_of_photon_count_at_mode(mode, modes, photons, device):
    """This function returs the list of orthogonal projectors that represent the photon-count
    measurement at the mode 'mode'. They are returned in increasing order of photon-count.
    Inputs:
        - mode: mode measured
        - modes: number of modes of the circuit
        - photons: number of photons in the circuit
    Output:
        - PVM_elements: list of 'photon'+1 orthogonal projectors
    """
    d = hilbert_dim(modes, photons)
    fock_basis = get_photon_basis(modes, photons)

    # Reminder: the way QOptCraft orders the Fock basis when it builds n-photon hamiltonians
    # (in its method `photon_hamiltonian` that I use) is the order of the Python list returned
    # by `get_photon_basis`. So this convention is respected for the PVM built here.

    PVM_elements = [
        torch.zeros((d, d)).type(torch.float).to(device) for _ in range(photons + 1)
    ]
    for i in range(d):
        k_of_this_basis_elem = fock_basis[i][mode]
        PVM_elements[k_of_this_basis_elem][i, i] = 1

    return PVM_elements


class PVM_Measurement(nn.Module):
    """This function defines a single PVM Measurement, equivalent to a measurement + generation
    of the number of photons measured."""

    def __init__(self, mode, modes, photons, device):
        super().__init__()
        self.mode = mode
        self.modes = modes
        self.photons = photons
        self.device = device
        self.PVM_elements = get_PVM_of_photon_count_at_mode(
            mode, modes, photons, device
        )

    def forward(self, rho):
        """This function returns the equivalent state after PVM measurement."""
        return sum(kraus_op @ rho @ kraus_op.conj() for kraus_op in self.PVM_elements)
