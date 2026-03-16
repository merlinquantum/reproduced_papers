"""
Bond Dimension Experiment Module

This module contains functions to run experiments evaluating the effect of different
bond dimensions on the performance of the Quantum Train.
"""

import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from papers.DQNN.lib.model import PhotonicQuantumTrain, train_quantum_model
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
    )
    from papers.DQNN.utils.utils import create_datasets, plot_bond_exp
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.DQNN.lib.model import PhotonicQuantumTrain, train_quantum_model
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
    )
    from papers.DQNN.utils.utils import create_datasets, plot_bond_exp

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_bond_dimension_exp(
    bond_dimensions_to_test: list[int] = np.arange(1, 11),
    num_training_rounds: int = 200,
    num_epochs: int = 5,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
    generate_graph: bool = True,
    run_dir: Path = None,
):
    """
    Run experiments to evaluate the impact of different bond dimensions on the performance of the Quantum Train.

    This function iterates over a list of bond dimensions, trains the quantum train, collects training loss and
    accuracy metrics, saves the results to a JSON file (/results/bond_dimension_data.json), and generates a plot
    comparing the performance across bond dimensions.

    Parameters
    -----------
    bond_dimensions_to_test : List[int], optional
        List of bond dimension values to test. Default is [1, 2, ..., 10].
    num_training_rounds : int, optional
        Number of training rounds (epochs) of MPS and quantum training. Default is 200.
    num_epochs : int, optional
        Number of epochs for per training round for the MPS. Default is 5.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA optimizer for quantum training. Default is False.
    num_qnn_train_step : int, optional
        Number of training steps for the boson samplers per training round. Default is 12.  If COBYLA
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
    current_dir = str(pathlib.Path(__file__).parent.parent.resolve()) + "/results/"
    losses = []
    accuracies = []

    # Bond dimension
    for bd in bond_dimensions_to_test:
        bs_1, bs_2 = create_boson_samplers()

        train_dataset, _, train_loader, _ = create_datasets()

        n_qubit, nw_list_normal = calculate_qubits()

        qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bd).to(device)

        batch_size_qnn = 1000
        train_loader_qnn = DataLoader(train_dataset, batch_size_qnn, shuffle=True)

        qt_model, _, loss_list_epoch, acc_list_epoch = train_quantum_model(
            qt_model,
            train_loader,
            train_loader_qnn,
            bs_1,
            bs_2,
            n_qubit,
            nw_list_normal,
            num_training_rounds=num_training_rounds,
            num_epochs=num_epochs,
            qu_train_with_cobyla=qu_train_with_cobyla,
            num_qnn_train_step=num_qnn_train_step,
        )
        losses.append(loss_list_epoch)
        accuracies.append(acc_list_epoch)

        json_str = json.dumps({"loss_list": losses, "acc_list": accuracies}, indent=4)
        with open(current_dir + "bond_dimension_data.json", "w") as f:
            f.write(json_str)
    if generate_graph:
        plot_bond_exp(
            bond_dimensions_to_test,
            np.arange(num_training_rounds),
            losses,
            accuracies,
            run_dir=run_dir,
        )


# run_bond_dimension_exp(
#    num_training_rounds=100, num_qnn_train_step=30, qu_train_with_cobyla=False
# )
