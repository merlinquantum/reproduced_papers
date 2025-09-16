"""
Definition of the actual models used to reproduce results from the paper
"""

import itertools
import math

import merlin
import torch
import torch.nn as nn

from photonic_QCNN.src.qcnn_paper import (
    Measure,
    OneHotEncoder,
    QConv2d,
    QDense,
    QPooling,
    generate_all_fock_states_list,
)


def marginalize_photon_presence(keys, probs):
    """
    Marginalize Fock state probabilities to get per-mode occupation probabilities.

    Computes the probability that each mode contains at least one photon
    by summing over all Fock states where that mode is occupied.

    Args:
        keys (list): List of Fock state tuples, e.g., [(0,1,0,2), (1,0,1,0), ...]
        probs (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
            for each Fock state, with requires_grad=True

    Returns:
        torch.Tensor: Shape (N, num_modes) with marginal probability that
            each mode has at least one photon
    """
    device = probs.device
    keys_tensor = torch.tensor(
        keys, dtype=torch.long, device=device
    )  # shape: (num_keys, num_modes)
    keys_tensor.shape[1]

    # Create mask of shape (num_modes, num_keys)
    # Each mask[i] is a binary vector indicating which Fock states have >=1 photon in mode i
    mask = (keys_tensor >= 1).T  # shape: (num_modes, num_keys)

    # Convert to float to allow matrix multiplication
    mask = mask.float()

    # Now do: (N, num_keys) @ (num_keys, num_modes) → (N, num_modes)
    marginalized = probs @ mask.T  # shape: (N, num_modes)
    return marginalized


def generate_partial_fock_states(subset, n, m):
    """
    Generate all the possible Fock state considering a subset of modes.

    Args:
    :param subset: Number of modes to consider. Has to be smaller or equal to m (number of modes)
    :param n: Number of photons
    :param m: Total number of modes
    :return: List of all possible Fock states considering the subset of modes
    """
    reduced_states = []
    # Account for when subset == m or subset + 1 == m. There cannot have 1 or 0 photon
    for i in range(max(0, subset - m + n), n + 1):
        reduced_states += generate_all_fock_states_list(subset, i, true_order=True)
    return reduced_states


def partial_measurement_output_size(subset: int, n: int, total_modes: int) -> int:
    """
    Compute number of possible measurement outcomes when measuring a subset
    of modes in Fock space, constrained by total photon number.

    Args:
        subset (int): Number of measured modes
        n (int): Total number of photons
        total_modes (int): Total number of modes (m)

    Returns:
        int: Number of reduced Fock states consistent with measurement
    """
    if subset == total_modes:
        # Full measurement: all photons must be in measured modes
        return math.comb(subset + n - 1, n)
    else:
        # Partial measurement: sum over all valid photon counts in measured modes
        return sum(math.comb(subset + i - 1, i) for i in range(n + 1))


class HybridModel(nn.Module):
    """
    Hybrid photonic quantum CNN model using MerLin framework.

    Combines quantum convolution, pooling, dense layers and measurement
    with classical post-processing for binary classification tasks.

    Args:
        dims (tuple): Input image dimensions (height, width)
        conv_circuit (str): Circuit type for convolution layer ('MZI', 'BS', etc.)
        dense_circuit (str): Circuit type for dense layer
        measure_subset (int): Number of modes to measure (None for all)
        dense_added_modes (int): Additional modes to insert in dense layer
        output_proba_type (str): 'state' for Fock state probabilities, 'mode' for per-mode
        output_formatting (str): Output mapping strategy ('Train_linear', 'Lex_grouping', etc.)
        num_classes (int): Number of output classes (default: 2)
    """

    def __init__(
        self,
        dims,
        conv_circuit,
        dense_circuit,
        measure_subset,
        dense_added_modes,
        output_proba_type,
        output_formatting,
        num_classes=2,
    ):
        super().__init__()
        self.num_modes_end = dims[0] + dense_added_modes
        self.num_modes_measured = (
            measure_subset if measure_subset is not None else dims[0]
        )

        self.one_hot_encoding = OneHotEncoder()
        self.conv2d = QConv2d(dims, kernel_size=2, stride=2, circuit=conv_circuit)
        self.pooling = QPooling(dims, kernel_size=2)
        self.dense = QDense(
            (int(dims[0] / 2), int(dims[1] / 2)),
            circuit=dense_circuit,
            add_modes=dense_added_modes,
        )
        self.measure = Measure(
            m=dims[0] + dense_added_modes, n=2, subset=measure_subset
        )

        self.qcnn = nn.Sequential(
            self.one_hot_encoding, self.conv2d, self.pooling, self.dense, self.measure
        )

        # Output dimension of the QCNN
        # Depends on whether we consider the probability of each Fock state or of each mode separately
        self.output_proba_type = output_proba_type
        if output_proba_type == "state":
            if measure_subset is not None:
                qcnn_output_dim = partial_measurement_output_size(
                    self.num_modes_measured, 2, self.num_modes_end
                )
            else:
                states = generate_all_fock_states_list(
                    self.num_modes_end, 2, true_order=True
                )
                qcnn_output_dim = len(states)
            print(f"Number of Fock states: {qcnn_output_dim}")

        elif output_proba_type == "mode":
            if measure_subset is not None:
                qcnn_output_dim = measure_subset
            else:
                qcnn_output_dim = self.num_modes_end  # Number of modes
        else:
            raise NotImplementedError(
                f"Output probability type {output_proba_type} not implemented"
            )
        self.qcnn_output_dim = qcnn_output_dim

        # Output mapping strategy
        if output_formatting == "Train_linear":
            self.output_mapping = nn.Linear(qcnn_output_dim, num_classes)
        elif output_formatting == "No_train_linear":
            self.output_mapping = nn.Linear(qcnn_output_dim, num_classes)
            self.output_mapping.weight.requires_grad = False
            self.output_mapping.bias.requires_grad = False
        elif output_formatting == "Lex_grouping":
            self.output_mapping = merlin.sampling.mappers.LexGroupingMapper(
                qcnn_output_dim, num_classes
            )
        elif output_formatting == "Mod_grouping":
            self.output_mapping = merlin.sampling.mappers.ModGroupingMapper(
                qcnn_output_dim, num_classes
            )
        else:
            raise NotImplementedError

        if measure_subset is not None:
            self.keys = generate_partial_fock_states(
                measure_subset, 2, self.num_modes_end
            )
        else:
            self.keys = generate_all_fock_states_list(
                self.num_modes_end, 2, true_order=True
            )

    def forward(self, x):
        probs = self.qcnn(x)

        if self.output_proba_type == "mode":
            probs = marginalize_photon_presence(self.keys, probs)

        output = self.output_mapping(probs)
        output = output * 66

        return output


class Readout(nn.Module):
    def __init__(self, list_label_0):
        super().__init__()
        self.list_label_0 = list_label_0
        self.list_label_1 = []
        self.initialize_labels()

    def forward(self, proba, keys):
        """
        proba: (batch_size, num_states)
        keys: list/tuple of mode indices corresponding to columns in proba
        """
        device = proba.device
        dtype = proba.dtype

        # Build masks for the two label groups
        mask_0 = torch.tensor(
            [k in self.list_label_0 for k in keys], device=device, dtype=dtype
        )  # shape (num_modes,)
        mask_1 = torch.tensor(
            [k in self.list_label_1 for k in keys], device=device, dtype=dtype
        )

        # Compute sums over the masked columns
        proba_0 = (proba * mask_0).sum(dim=1)  # shape (batch_size,)
        proba_1 = (proba * mask_1).sum(dim=1)

        total = proba_0 + proba_1
        out = torch.stack([proba_0, proba_1], dim=1) / total.unsqueeze(
            1
        )  # shape (batch_size, 2)

        return out

    def initialize_labels(self):
        modes = list(range(6))  # modes 0, 1, 2, 3, 4, 5
        pairs = list(itertools.combinations(modes, 2))  # 15 of them
        binary_pairs = []
        for i, j in pairs:
            vec = [0] * 6
            vec[i] = 1
            vec[j] = 1
            binary_pairs.append(tuple(vec))

        for binary_pair in binary_pairs:
            if binary_pair not in self.list_label_0:
                self.list_label_1.append(binary_pair)
        return


class HybridModelReadout(nn.Module):
    """
    Hybrid photonic quantum CNN model using MerLin framework utilizing the two strategies to configure the measurement
    layer:
    1. Associate every two-mode configuration to a label and the two modes with the photons determines which label
    is associated.
    2. Associate a single pair of modes to label 0 and the rest to label 1.

    Combines quantum convolution, pooling, dense layers and measurement
    with first readout strategy for binary classification tasks.

    Args:
        dims (tuple): Input image dimensions (height, width)
        conv_circuit (str): Circuit type for convolution layer ('MZI', 'BS', etc.)
        dense_circuit (str): Circuit type for dense layer
        dense_added_modes (int): Additional modes to insert in dense layer
        num_classes (int): Number of output classes (default: 2)
        list_label_0 (list): List of photon configurations associated with label 0
    """

    def __init__(
        self,
        dims,
        conv_circuit,
        dense_circuit,
        dense_added_modes,
        list_label_0,
        num_classes=2,
    ):
        super().__init__()
        self.num_modes_end = dims[0] + dense_added_modes
        self.num_modes_measured = dims[0]

        self.one_hot_encoding = OneHotEncoder()
        self.conv2d = QConv2d(dims, kernel_size=2, stride=2, circuit=conv_circuit)
        self.pooling = QPooling(dims, kernel_size=2)
        self.dense = QDense(
            (int(dims[0] / 2), int(dims[1] / 2)),
            circuit=dense_circuit,
            add_modes=dense_added_modes,
        )
        self.measure = Measure(m=dims[0] + dense_added_modes, n=2, subset=None)

        self.qcnn = nn.Sequential(
            self.one_hot_encoding, self.conv2d, self.pooling, self.dense, self.measure
        )

        # Output dimension of the QCNN
        states = generate_all_fock_states_list(self.num_modes_end, 2, true_order=True)
        qcnn_output_dim = len(states)
        print(f"Number of Fock states: {qcnn_output_dim}")
        self.qcnn_output_dim = qcnn_output_dim

        self.readout = Readout(list_label_0)

        self.keys = generate_all_fock_states_list(
            self.num_modes_end, 2, true_order=True
        )

    def forward(self, x):
        probs = self.qcnn(x)

        output = self.readout(probs, self.keys)
        output = output * 66

        return output
