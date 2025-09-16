# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
from qoptcraft.evolution import *
from qoptcraft.optical_elements import *
from torch import nn

from photonic_QCNN.src.paper_layers.toolbox import *


#####################################################################################################################
### BeamSplitter in the 2-dimensional image subspace:                                                             ###
#####################################################################################################################
def SinglePhoton_BS_Unitary(d1, d2, gate_impact, device):
    """Return an RBS corresponding unitary decomposed as coeffs that should be multiplied by
    cos(theta), and coeffs that should be multiplied by sin(theta). This decomposition allows to avoid inplace operations.
    Inputs:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - gate_impact: list of tuples of basis vectors. Their planar rotation satisfies
        this transformation
    """
    id_matrix = torch.eye((d1 * d2), dtype=torch.float32, device=device)
    cos_matrix = torch.zeros(((d1 * d2), (d1 * d2)), dtype=torch.float32, device=device)
    sin_matrix = torch.zeros(((d1 * d2), (d1 * d2)), dtype=torch.float32, device=device)
    # Adapting the gates to the number of modes:
    for state_1, state_2 in gate_impact:
        id_matrix[state_1, state_1] = 0
        id_matrix[state_2, state_2] = 0
        sin_matrix[state_1, state_1] = 1
        sin_matrix[state_2, state_2] = 1
        cos_matrix[state_1, state_2] = 1
        cos_matrix[state_2, state_1] = -1
    return (cos_matrix, sin_matrix, id_matrix)


class BS_Registers_density(nn.Module):
    """This module describes the action of a quantum layer with BS gates. We consider
    two distinct and independent registers of size d1 and d2."""

    def __init__(self, d1, d2, gate_impact, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - gate_impact: list of tuples of basis vectors. Their planar rotation satisfies
        this transformation
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.device = device
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)
        self.cos_matrix, self.sin_matrix, self.id_matrix = SinglePhoton_BS_Unitary(
            d1, d2, gate_impact, device
        )

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        return (
            (
                self.cos_matrix * torch.cos(self.angle)
                + self.sin_matrix * torch.sin(self.angle)
                + self.id_matrix
            )
            .matmul(input_state)
            .matmul(
                (
                    self.cos_matrix * torch.cos(self.angle)
                    + self.sin_matrix * torch.sin(self.angle)
                    + self.id_matrix
                ).t()
            )
        )


def Conv_BS_mode_states(d1, d2, m1, m2):
    """Return the list of the states impacted by a gate between two modes in a register.
    Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - m1: first mode
        - m2: second mode
    """
    list_state_tuples = []
    if (m1 < d1) and (m2 < d1):  # Both modes are in the first register
        mode1, mode2 = m1, m2
        for i in range(d2):
            list_state_tuples.append((mode1 * d2 + i, mode2 * d2 + i))
    elif (m1 >= d1) and (m2 >= d1):  # Both modes are in the second register
        mode1, mode2 = m1 - d1, m2 - d1
        for i in range(d1):
            list_state_tuples.append((i * d2 + mode1, i * d2 + mode2))
    return list_state_tuples


class VQC_Registers_BS_density(nn.Module):
    """This module describes the action of a quantum layer with BS gates. We consider
    two distinct and independent registers of size d1 and d2."""

    def __init__(self, d1, d2, list_gates, device):
        """Args:
        - d1: number of modes in the first register
        - d2: number of modes in the second register
        - list_gates: list of list of tuples (m1,m2) representing the modes affected by each RBS.
        Each list represent a particular parameter. m1 is in the first register and m2 in the
        second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        list_gate_impacts = [[] for i in range(len(list_gates))]
        for i, gates_one_parameter in enumerate(list_gates):
            for m1, m2 in gates_one_parameter:
                list_gate_impacts[i] += Conv_BS_mode_states(d1, d2, m1, m2)
        self.VQC = nn.ModuleList(
            BS_Registers_density(d1, d2, gates_impact, device)
            for gates_impact in list_gate_impacts
        )

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for BS in self.VQC:
            input_state = BS(input_state)
        return input_state


#####################################################################################################################
### PhaseShifter in the Fock Basis:                                                                               ###
#####################################################################################################################
def Phase_Flip_Fock_Unitary(n, m, mode1, device):
    """This class describes the action of a single PhaseShifter on a Fock basis state.
    Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - mode1: mode of the phase shifter
        - device: torch device (cpu, cuda, etc...)
    Output:
        - unitary matrix of the phase shifter in the Fock basis
    """
    shift = torch.pi  # parameter to use for the phase shifter to perform a phase flip
    PS = phase_shifter(shift, m, mode1)
    if n == 1:
        return torch.from_numpy(PS).real
    else:
        return torch.from_numpy(photon_unitary(PS, n)).real


class PhaseFlip_Fock_density(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state."""

    def __init__(self, n, m, mode1, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - mode1: mode of the phase shifter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.U = Phase_Flip_Fock_Unitary(n, m, mode1, device).type(torch.float32)

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        return (self.U).matmul(input_state).matmul((self.U).t())


class PS_Fock_unitary(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state
    and in the unitary space."""

    def __init__(self, n, m, mode1, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - mode1: mode of the phase shifter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.Unitary = Phase_Flip_Fock_Unitary(n, m, mode1, device).type(torch.float32)

    def forward(self, matrix):
        return self.Unitary.matmul(matrix)


#####################################################################################################################
### BeamSplitter in the Fock Basis:                                                                               ###
#####################################################################################################################
def SinglePhotonHamiltonianBS(modes=4, mode_1=0, mode_2=1):
    """This function returns the Hamiltonian of a beamsplitter considering only one photon.
    Inputs:
        - modes: number of modes of the system
        - mode_1: first mode of the beamsplitter
        - mode_2: second mode of the beamsplitter
    Output:
        - H: Hamiltonian of the beamsplitter
    """
    H = torch.zeros((modes, modes), dtype=torch.double)
    H[mode_1, mode_2] = -1
    H[mode_2, mode_1] = +1
    return H


def HamiltonianBS(photons=2, modes=4, mode_1=0, mode_2=1, device="cpu"):
    """This function returns the Hamiltonian of a beamsplitter considering several photons.
    Inputs:
        - photons: number of photons of the system
        - modes: number of modes of the system
        - mode_1: first mode of the beamsplitter
        - mode_2: second mode of the beamsplitter
    Output:
        - H: Hamiltonian of the beamsplitter
    """
    HS = SinglePhotonHamiltonianBS(modes, mode_1, mode_2)
    HU = photon_hamiltonian(HS, photons)
    HU = torch.tensor(HU, dtype=torch.float32, device=device)
    return HU


class BS_Fock_state_vector(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state."""

    def __init__(self, n, m, m1, m2, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial state vector on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        torch.matrix_exp(-1 * self.H * self.angle)
        return (torch.matrix_exp(-1 * self.H * self.angle)).matmul(input_state)


class BS_Fock_density(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state."""

    def __init__(self, n, m, m1, m2, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        torch.matrix_exp(-1 * self.H * self.angle)
        return (
            (torch.matrix_exp(-1 * self.H * self.angle))
            .matmul(input_state)
            .matmul((torch.matrix_exp(-1 * self.H * self.angle)).t())
        )


class BS_Fock_density_Parameter(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state."""

    def __init__(self, n, m, m1, m2, param, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter
        - param: angle of the beamsplitter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.angle = nn.Parameter(param, requires_grad=False)

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        torch.matrix_exp(-1 * self.H * self.angle)
        return (
            (torch.matrix_exp(-1 * self.H * self.angle))
            .matmul(input_state)
            .matmul((torch.matrix_exp(-1 * self.H * self.angle)).t())
        )


class BS_Fock_PureState(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state."""

    def __init__(self, n, m, m1, m2, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial input state on which is applied the RBS from
            the VQC
        Output:
            - final state matrix from the application of the RBS from the VQC on the
            input state
        """
        torch.matrix_exp(-1 * self.H * self.angle)
        return (torch.matrix_exp(-1 * self.H * self.angle)).matmul(input_state)


class BS_Fock_unitary(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state
    and in the unitary space."""

    def __init__(self, n, m, m1, m2, param, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter
        - param: angle of the beamsplitter
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.angle = param
        self.Unitary = torch.matrix_exp(-1 * self.H * self.angle).to(device)

    def forward(self, matrix):
        return self.Unitary.matmul(matrix)


class SWAP_Fock_unitary(nn.Module):
    """This class describes the action of a single BeamSplitter on a Fock basis state
    and in the unitary space."""

    def __init__(self, n, m, m1, m2, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - m1: first mode of the beamsplitter
        - m2: second mode of the beamsplitter (m1 < m2)
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.H_BS = HamiltonianBS(n, m, m1, m2, device).type(torch.float32)
        self.Unitary_BS = torch.matrix_exp(-1 * self.H_BS * torch.tensor(torch.pi / 2))
        self.Unitary_PS = PS_Fock_unitary(
            n, m, m2, device
        )  # Phase shifter on the second mode

    def forward(self, matrix):
        return self.Unitary_PS(self.Unitary_BS.matmul(matrix))


class VQC_Fock_BS_state_vector(nn.Module):
    """This module describes the action of a quantum layer with BS gates."""

    def __init__(self, n, m, list_gates, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - list_gates: list of tuples (m1,m2) representing the modes affected by each RBS.
        m1 is in the first register and m2 in the second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.VQC = nn.ModuleList(
            BS_Fock_state_vector(n, m, m1, m2, device) for (m1, m2) in list_gates
        )

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for BS in self.VQC:
            input_state = BS(input_state)
        return input_state


class VQC_Fock_BS_density(nn.Module):
    """This module describes the action of a quantum layer with BS gates."""

    def __init__(self, n, m, list_gates, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - list_gates: list of tuples (m1,m2) representing the modes affected by each RBS.
        m1 is in the first register and m2 in the second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.VQC = nn.ModuleList(
            BS_Fock_density(n, m, m1, m2, device) for (m1, m2) in list_gates
        )

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for BS in self.VQC:
            input_state = BS(input_state)
        return input_state


class VQC_Fock_BS_density_Parameters(nn.Module):
    """This module describes the action of a quantum layer with BS gates."""

    def __init__(self, n, m, list_gates, list_parameters, device):
        """Args:
        - n: number of photons in the system
        - m: number of modes in the system
        - list_gates: list of tuples (m1,m2) representing the modes affected by each RBS.
        m1 is in the first register and m2 in the second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.VQC = nn.ModuleList(
            BS_Fock_density_Parameter(n, m, m1, m2, list_parameters[i], device)
            for i, (m1, m2) in enumerate(list_gates)
        )

    def forward(self, input_state):
        """Feedforward of the convolutional layer with BS gates.
        Arg:
            - input_state = a initial density matrix on which is applied the RBS from
            the VQC
        Output:
            - final density matrix from the application of the RBS from the VQC on the
            input state
        """
        for BS in self.VQC:
            input_state = BS(input_state)
        return input_state


class Equivalent_Unitary_BS_Fock(nn.Module):
    """This module describes the action of a quantum layer with BS gates
    in the unitary space."""

    def __init__(self, n, m, list_gates, list_param, device):
        """Args:
        - n: number of photon in the considered subspace
        - m: number of modes in the system
        - list_gates: list of tuples (m1,m2) representing the modes affected by each BS.
        m1 is in the first register and m2 in the second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.VQC = nn.ModuleList(
            BS_Fock_unitary(n, m, m1, m2, list_param[i], device)
            for i, (m1, m2) in enumerate(list_gates)
        )

    def forward(self, matrix):
        for BS in self.VQC:
            matrix = BS(matrix)
        return matrix


class Equivalent_Unitary_SWAP_Fock(nn.Module):
    """This module describes the action of a quantum layer with SWAP gates
    in the unitary space."""

    def __init__(self, n, m, list_gates, device):
        """Args:
        - n: number of photon in the considered subspace
        - m: number of modes in the system
        - list_gates: list of tuples (m1,m2) representing the modes affected by each SWAP.
        m1 is in the first register and m2 in the second register.
        - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.VQC = nn.ModuleList(
            SWAP_Fock_unitary(n, m, m1, m2, device) for (m1, m2) in list_gates
        )

    def forward(self, matrix):
        for SWAP in self.VQC:
            matrix = SWAP(matrix)
        return matrix
