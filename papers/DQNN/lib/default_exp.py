"""
Default experiment runner for the DQNN photonic quantum train model.

This module trains a classical CNN baseline, then trains and evaluates the
photonic quantum train model using the specified hyperparameters.
"""

import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

try:
    from papers.DQNN.lib.classical_utils import (
        train_classical_cnn,
    )
    from papers.DQNN.lib.model import (
        PhotonicQuantumTrain,
        evaluate_model,
        train_quantum_model,
    )
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
    )
    from papers.DQNN.utils.utils import create_datasets, plot_training_metrics
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.DQNN.lib.classical_utils import (
        train_classical_cnn,
    )
    from papers.DQNN.lib.model import (
        PhotonicQuantumTrain,
        evaluate_model,
        train_quantum_model,
    )
    from papers.DQNN.lib.photonic_qt_utils import (
        calculate_qubits,
        create_boson_samplers,
    )
    from papers.DQNN.utils.utils import create_datasets, plot_training_metrics

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_default_exp(
    bond_dim: int = 7,
    num_training_rounds: int = 50,
    num_epochs: int = 5,
    num_qnn_train_step: int = 12,
    classical_epochs: int = 5,
    pruning: bool = False,
    pruning_amount: float = 0.5,
    weight_sharing: bool = False,
    shared_rows: int = 10,
    qu_train_with_cobyla: bool = False,
    generate_graph: bool = True,
    run_dir: Path = None,
):
    """
    Run the default experiment workflow.

    Parameters
    ----------
    bond_dim : int, optional
        Bond dimension for the photonic quantum train model. Default is 7.
    num_training_rounds : int, optional
        Number training rounds: the number of MPS and then the quantum layer training epochs. Default is 50.
    num_epochs : int, optional
        Number of epochs per training round for the MPS. Default is 5.
    num_qnn_train_step : int, optional
        Number of QNN training steps per round. Default is 12.
    classical_epochs : int, optional
        Number of epochs for the classical CNN baseline. Default is 5.
    pruning : bool, optional
        Whether to enable pruning during classical training. Default is False.
    pruning_amount : float, optional
        Fraction of weights to prune if pruning is enabled. Default is 0.5.
    weight_sharing : bool, optional
        Whether to enable weight sharing in the classical model. Default is False.
    shared_rows : int, optional
        Number of rows to share if weight sharing is enabled. Default is 10.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA optimizer for quantum training. Default is False. If COBYLA
        is to be used, 1000 is the suggested value.
    generate_graph : bool, optional
        Whether to plot a summary of train/test metrics after evaluation.
        Default is True.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    -------
    None
    """

    print("Running experiment with:")
    print(f"  Pruning: {pruning} (amount: {pruning_amount if pruning else 'N/A'})")
    print(
        f"  Weight sharing: {weight_sharing} (shared rows: {shared_rows if weight_sharing else 'N/A'})"
    )

    bs_1, bs_2 = create_boson_samplers()

    train_dataset, _, train_loader, val_loader = create_datasets()

    _ = train_classical_cnn(
        train_loader,
        val_loader,
        classical_epochs,
        use_pruning=pruning,
        pruning_amount=pruning_amount,
        use_weight_sharing=weight_sharing,
        shared_rows=shared_rows,
    )

    n_qubit, nw_list_normal = calculate_qubits()

    qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bond_dim).to(device)

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
        num_training_rounds,
        num_epochs,
        qu_train_with_cobyla=qu_train_with_cobyla,
        num_qnn_train_step=num_qnn_train_step,
    )

    evaluate_model(
        qt_model,
        train_loader,
        val_loader,
        bs_1,
        bs_2,
        n_qubit,
        nw_list_normal,
    )

    if generate_graph:
        plot_training_metrics(loss_list_epoch, acc_list_epoch, run_dir=run_dir)
