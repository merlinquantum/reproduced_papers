"""
Photonic quantum train utilities for DQNN experiments.

This module provides helpers for boson sampler creation,
qubit calculations, and probability-to-weight mapping utilities.
"""

import numpy as np
import torch

from papers.DQNN.lib.boson_sampler import BosonSampler

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def create_boson_samplers() -> BosonSampler:
    """
    Create the boson samplers used in photonic quantum training.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        Two BosonSampler instances configured for the experiment.
    """
    bs_1 = BosonSampler(m=9, n=4)
    print(
        f"Boson sampler defined with number of parameters = {bs_1.nb_parameters}, and embedding size = {bs_1.embedding_size}"
    )

    bs_2 = BosonSampler(m=8, n=4)
    print(
        f"Boson sampler defined with number of parameters = {bs_2.nb_parameters}, and embedding size = {bs_2.embedding_size}"
    )
    return bs_1, bs_2


def calculate_qubits() -> tuple[int, list[float]]:
    """
    Compute the number of qubits required for the CNN weight mapping.

    Returns
    -------
    Tuple[int, List[float]]
        Number of qubits and flattened list of CNN parameters.
    """
    from papers.DQNN.lib.classical_utils import CNNModel

    standard_model = CNNModel(use_weight_sharing=False, shared_rows=10)

    numpy_weights = {}
    nw_list = []
    nw_list_normal = []
    for name, param in standard_model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters for quantum circuit: ", len(nw_list_normal))
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)
    n_qubit = n_qubits
    return n_qubit, nw_list_normal


def probs_to_weights(probs_: torch.Tensor, model_template: torch.nn.Module) -> dict:
    """
    Convert a flat probability tensor into a model state dict.

    Parameters
    ----------
    probs_ : torch.Tensor
        Flattened probability values.
    model_template : torch.nn.Module
        Model whose state dict shapes are used for reshaping.

    Returns
    -------
    dict
        State dict with tensors reshaped to match the template model.
    """
    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in model_template.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]

    return new_state_dict


def generate_qubit_states_torch(n_qubit: int) -> torch.Tensor:
    """
    Generate all qubit states in the {-1, 1} basis.

    Parameters
    ----------
    n_qubit : int
        Number of qubits.

    Returns
    -------
    torch.Tensor
        Tensor of shape (2**n_qubit, n_qubit) with all basis states.
    """
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states
