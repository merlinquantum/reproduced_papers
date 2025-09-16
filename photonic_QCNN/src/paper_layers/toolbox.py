# ruff: noqa: N801, N806, N802, N803, E741, F405, F403
import torch
from qoptcraft.basis import *


def PQNN(m):
    """This function defines the different gates that form
    a PQNN circuit.
    Args:
        - m: number of modes
    Outputs:
        - gates: list of gates (list of tuples (m1,m2))
    """
    gates = []
    for i in range(1, m):
        gates.append((i - 1, i))
        k = i - 2
        while k > 0:
            gates.append((k - 1, k))
            k = k - 2
    for i in range(m - 2, 0, -1):
        gates.append((i - 1, i))
        k = i - 2
        while k > 0:
            gates.append((k - 1, k))
            k = k - 2
    return gates


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


def Pyramidal_Order_RBS_gates(nbr_qubits, first_RBS=0):
    """This function gives the structure of each inner layer in the pyramidal
    quantum neural network. List_order gives the qubit link to each theta and
    List_layer_index gives the list of the theta for each inner layer."""
    List_layers, List_order, List_layer_index = [], [], []
    index_RBS = first_RBS
    # Beginning of the pyramid
    for i in range(nbr_qubits // 2):
        list, list_index = [], []
        for j in range(i + 1):
            if i * 2 < (nbr_qubits - 1):
                list.append(j * 2)
                list_index.append(index_RBS)
                index_RBS += 1
        if len(list) > 0:
            List_layers.append(list)
            List_layer_index.append(list_index)
        list, list_index = [], []
        for j in range(i + 1):
            if i * 2 + 1 < (nbr_qubits - 1):
                list.append(j * 2 + 1)
                list_index.append(index_RBS)
                index_RBS += 1
        if len(list) > 0:
            List_layers.append(list)
            List_layer_index.append(list_index)
    # End of the pyramid
    for i in range(len(List_layers) - 2, -1, -1):
        List_layers.append(List_layers[i])
        list_index = []
        for _j in range(len(List_layers[i])):
            list_index.append(index_RBS)
            index_RBS += 1
        List_layer_index.append(list_index)
    # Deconcatenate:
    for _i, layer in enumerate(List_layers):
        List_order += layer
    return (List_order, List_layer_index)


def Unitary_to_Pure_State(unitary_matrix, input):
    """This function returns the quantum channel corresponding to the unitary matrix."""
    input = input.unsqueeze(-1)
    unitary_matrix1 = torch.zeros([4, 126, 126])
    for i in range(4):
        unitary_matrix1[i] = unitary_matrix

    # print("in unitary to pure state",input.size(), unitary_matrix1.size())
    input = unitary_matrix1.bmm(input)
    input = input.squeeze(-1)
    # print(input.size())
    return input


def compute_jacobian(model, input_tensor):
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    num_outputs = output.numel()
    num_params = sum(p.numel() for p in model.parameters())

    jacobian_matrix = torch.zeros(num_outputs, num_params)

    param_idx = 0
    for param in model.parameters():
        param_size = param.numel()

        for i in range(param_size):
            gradient_output_i = torch.zeros_like(output.view(-1))
            gradient_output_i[i] = 1.0

            model.zero_grad()
            output.backward(gradient=gradient_output_i, retain_graph=True)
            jacobian_matrix[:, param_idx] = input_tensor.grad.view(-1)

            param_idx += 1

    input_tensor.requires_grad_(False)
    return jacobian_matrix
