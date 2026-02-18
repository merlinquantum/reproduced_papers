"""
Ablation Experiment Module

This module contains functions to run experiments evaluating the importance
of the quantum layer in the Quantum Train algorithm.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from papers.DQNN.lib.model import (
        PhotonicQuantumTrain,
        evaluate_model,
        train_quantum_model,
    )
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
        generate_qubit_states_torch,
        probs_to_weights,
    )
    from papers.DQNN.lib.torchmps import MPS
    from papers.DQNN.utils.utils import create_datasets, plot_ablation_exp
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.DQNN.lib.model import (
        PhotonicQuantumTrain,
        evaluate_model,
        train_quantum_model,
    )
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
        generate_qubit_states_torch,
        probs_to_weights,
    )
    from papers.DQNN.lib.torchmps import MPS
    from papers.DQNN.utils.utils import create_datasets, plot_ablation_exp

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def create_ablation_class(bond):
    """
    Create an lone MPS model for the experiment.

    This function defines a CNN model and an ablation module that mimics the quantum train behavior
    using only the classical MPS, along with data loaders for training and validation.

    Parameters
    -----------
    bond : int
        The bond dimension for the MPS in the ablation module.

    Returns
    --------
    tuple
        A tuple containing the ablation model (AblationModule)
    """

    n_qubit, nw_list_normal = calculate_qubits()

    random_tensor = torch.randn(
        126 * 70, 1
    )  # TODO FOr generalization, change to arbitrary size, here is the number of combinations of each BS (4 in 9) and (4 in 8)

    class AblationModule(nn.Module):
        """
        Ablation study module that uses MPS to process weights instead of quantum boson samplers.
        """

        def __init__(self):
            """
            Initialize the AblationModule with an MPS for weight processing.
            """
            super().__init__()

            self.MappingNetwork = MPS(
                input_dim=n_qubit + 1, output_dim=1, bond_dim=bond
            )

        def forward(self, x):
            """
            Forward pass through the ablation module.

            This method processes the input through a classical MPS-based weight generation
            and then applies the CNN layers.

            Parameters
            -----------
            x : torch.Tensor
                Input tensor of shape (batch_size, 1, 28, 28).

            Returns
            --------
            torch.Tensor
                Output tensor of shape (batch_size, 10).
            """
            from papers.DQNN.lib.classical_utils import CNNModel

            probs_ = random_tensor

            probs_ = probs_[: len(nw_list_normal)]
            probs_ = probs_.reshape(len(nw_list_normal), 1)

            # Generate qubit states using PyTorch
            qubit_states_torch = generate_qubit_states_torch(n_qubit)[
                : len(nw_list_normal)
            ]
            qubit_states_torch = qubit_states_torch.to(device)

            # Combine qubit states with probability values using PyTorch
            combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
            combined_data_torch = combined_data_torch.reshape(
                len(nw_list_normal), n_qubit + 1
            )

            prob_val_post_processed = self.MappingNetwork(combined_data_torch)
            prob_val_post_processed = (
                prob_val_post_processed - prob_val_post_processed.mean()
            )

            model_template = CNNModel(use_weight_sharing=False, shared_rows=10)
            state_dict = probs_to_weights(prob_val_post_processed, model_template)

            ########

            dtype = torch.float32  # Ensure all tensors are of this type

            # Convolution layer 1 parameters
            conv1_weight = state_dict["conv1.weight"].to(device).type(dtype)
            conv1_bias = state_dict["conv1.bias"].to(device).type(dtype)

            # Convolution layer 2 parameters
            conv2_weight = state_dict["conv2.weight"].to(device).type(dtype)
            conv2_bias = state_dict["conv2.bias"].to(device).type(dtype)

            # Fully connected layer 1 parameters
            fc1_weight = state_dict["fc1.weight"].to(device).type(dtype)
            fc1_bias = state_dict["fc1.bias"].to(device).type(dtype)

            # Fully connected layer 2 parameters
            fc2_weight = state_dict["fc2.weight"].to(device).type(dtype)
            fc2_bias = state_dict["fc2.bias"].to(device).type(dtype)

            # Convolution 1
            x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            # Convolution 2
            x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully connected 1
            x = F.linear(x, fc1_weight, fc1_bias)

            # Fully connected 2
            x = F.linear(x, fc2_weight, fc2_bias)

            return x

    return AblationModule().to(device)


def evaluate_ab_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
):
    """
    Evaluate the ablation model on a validation loader.

    Parameters
    ----------
    model : torch.nn.Module
        Ablation model to evaluate.
    val_loader : DataLoader
        Validation data loader.

    Returns
    -------
    tuple[float, float]
        Accuracy in percent and mean validation loss.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    loss_test_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_test = criterion(outputs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
    print(f"Loss on the test set: {np.mean(loss_test_list):.2f}")

    return (
        100 * correct / total,
        np.mean(loss_test_list, dtype=float),
    )


def run_ablation_exp(
    bond_dimensions_to_test: list[int] = np.arange(2, 17),
    num_training_rounds: int = 2,
    num_epochs: int = 5,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
    generate_graph: bool = True,
    run_dir: Path = None,
):
    """
    Run ablation experiments to evaluate the importance of the quantum layer in the Quantum Train.

    This function iterates over a list of bond dimensions, trains both the photonic QT and an ablation
    study model, collects training loss and accuracy metrics, saves the results to a JSON file
    (/results/ablation_data.json), and can generate a plot comparing the performance across models.

    Parameters
    -----------
    bond_dimensions_to_test : List[int], optional
        List of bond dimension values to test. Default is [2, 3, ..., 16].
    num_training_rounds : int, optional
        Number of training rounds (epochs) of MPS and quantum training. Default is 200.
    num_epochs : int, optional
        Number of epochs for per training round for the MPS. Default is 5.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA optimizer for quantum training. Default is False.
    num_qnn_train_step : int, optional
        Number of training steps for the boson samplers per training round. Default is 12. If COBYLA
        is to be used, 1000 is the suggested value.
    generate_graph : bool, optional
        Whether to plot a the resulting graph of the experiment.
        Default is True.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    --------
    None
    """
    current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"

    loss_ablation = []
    accuracy_ablation = []
    params_ablation = []
    loss_qt = []
    accuracy_qt = []
    params_qt = []

    _, _, train_loader, val_loader = create_datasets(batch_size=1000)

    for bond in bond_dimensions_to_test:
        ### QTrain
        bs_1, bs_2 = create_boson_samplers()
        n_qubit, nw_list_normal = calculate_qubits()
        qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bond).to(device)

        ### Ablation
        ablation_model = create_ablation_class(bond=bond)

        params_ablation.append(
            sum(p.numel() for p in ablation_model.parameters() if p.requires_grad)
        )

        # Training setting
        step = 1e-3  # Learning rate
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(ablation_model.parameters(), lr=step)

        # Training loop for the ablation
        for round_ in range(num_training_rounds):
            print("-----------------------")
            print("Ablation")

            acc_list = []
            acc_best = 0
            for epoch in range(num_epochs):
                ablation_model.train()
                train_loss = 0
                for i, (images, labels) in enumerate(train_loader):
                    correct = 0
                    total = 0
                    since_batch = time.time()

                    images, labels = (
                        images.to(device),
                        labels.to(device),
                    )  # Move data to GPU
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = ablation_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Compute loss
                    loss = criterion(outputs, labels)

                    acc = 100 * correct / total
                    acc_list.append(acc)
                    train_loss += loss.cpu().detach().numpy()

                    if acc > acc_best:
                        acc_best = acc
                    loss.backward()

                    optimizer.step()
                    if (i + 1) * 4 % len(train_loader) == 0:
                        print(
                            f"Training round [{round_ + 1}/{num_training_rounds}], Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                        )
                    loss_ab = loss.item()
                    acc_ab = acc

                train_loss /= len(train_loader)

        acc_ab, loss_ab = evaluate_ab_model(ablation_model, val_loader)

        ################################################################################################################################
        print("QTrain")

        qt_model, qnn_parameters_qt, _, _ = train_quantum_model(
            qt_model,
            train_loader,
            train_loader,
            bs_1,
            bs_2,
            n_qubit,
            nw_list_normal,
            num_training_rounds=num_training_rounds,
            num_epochs=num_epochs,
            qu_train_with_cobyla=qu_train_with_cobyla,
            num_qnn_train_step=num_qnn_train_step,
        )

        accuracy_test, loss_test, _ = evaluate_model(
            qt_model,
            train_loader,
            val_loader,
            bs_1,
            bs_2,
            n_qubit,
            nw_list_normal,
            qnn_parameters_qt,
        )

        num_trainable_params_qt = sum(
            p.numel() for p in qt_model.parameters() if p.requires_grad
        )

        # Save results
        params_qt.append(
            num_trainable_params_qt
            + bs_1.num_effective_params
            + bs_2.num_effective_params
        )
        loss_qt.append(loss_test)
        accuracy_qt.append(accuracy_test)

        loss_ablation.append(loss_ab)
        accuracy_ablation.append(acc_ab)

        json_payload = {
            "loss_qt": [float(v) for v in loss_qt],
            "accuracy_qt": [float(v) for v in accuracy_qt],
            "params_qt": [int(v) for v in params_qt],
            "loss_ablation": [float(v) for v in loss_ablation],
            "accuracy_ablation": [float(v) for v in accuracy_ablation],
            "params_ablation": [int(v) for v in params_ablation],
        }
        json_str = json.dumps(json_payload, indent=4)
        with open(current_dir + "ablation_data.json", "w") as f:
            f.write(json_str)
    if generate_graph:
        plot_ablation_exp(
            params_qt=params_qt,
            accuracy_qt=accuracy_qt,
            params_ablation=params_ablation,
            accuracy_ablation=accuracy_ablation,
            run_dir=run_dir,
        )


# run_ablation_exp(num_training_rounds=50)
