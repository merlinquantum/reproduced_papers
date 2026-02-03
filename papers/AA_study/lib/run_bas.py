import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.lib.qiskit_models import qiskit_QCNN
from papers.AA_study.lib.qlayers import PhotonicQCNN
from papers.AA_study.lib.classical_models import CNN
from papers.AA_study.utils.datasets import get_bas, get_data_loader
from papers.AA_study.utils.plots import plot_bas_run
from papers.AA_study.utils.utils import basic_model_training, evaluate_model


def run_bas(
    batch_size: int = 50,
    num_epochs: int = 20,
    classical_epochs: int = 20,
    lr: float = 0.01,
    run_dir: Path = None,
    generate_graph: bool = False,
):
    train_dataset, test_dataset = get_bas()
    train_loader = get_data_loader(train_dataset, batch_size=batch_size)
    test_loader = get_data_loader(test_dataset, batch_size=200)

    qiskit_model = qiskit_QCNN(num_qubits=4, num_classes=2)
    merlin_model = PhotonicQCNN(
        dims=(4, 4),
        conv_circuit="MZI",
        dense_circuit="MZI",
        measure_subset=None,
        dense_added_modes=0,
        output_proba_type="state",
        output_formatting="Lex_grouping",
        num_classes=2,
    )
    classical_model = CNN(input_image_size=4, num_layers=1, kernel_size=2)

    print("Classical model:")
    (classical_model, classical_model_accuracies, classical_model_losses) = (
        basic_model_training(
            classical_model, train_loader, lr=lr, num_epochs=classical_epochs
        )
    )
    classical_model_final_loss, classical_model_final_accuracy = evaluate_model(
        classical_model, test_loader
    )

    print("Qiskit model:")
    qiskit_model, qiskit_model_accuracies, qiskit_model_losses = basic_model_training(
        qiskit_model, train_loader, lr=lr, num_epochs=num_epochs
    )
    qiskit_model_final_loss, qiskit_model_final_accuracy = evaluate_model(
        qiskit_model, test_loader
    )

    print("MerLin model:")
    merlin_model, merlin_model_accuracies, merlin_model_losses = basic_model_training(
        merlin_model, train_loader, lr=lr, num_epochs=num_epochs
    )
    merlin_model_final_loss, merlin_model_final_accuracy = evaluate_model(
        merlin_model, test_loader
    )

    json_payload = {
        "classical_model_accuracies": [float(v) for v in classical_model_accuracies],
        "classical_model_losses": [float(v) for v in classical_model_losses],
        "classical_model_final_loss": classical_model_final_loss,
        "classical_model_final_accuracy": classical_model_final_accuracy,
        "qiskit_model_accuracies": [float(v) for v in qiskit_model_accuracies],
        "qiskit_model_losses": [float(v) for v in qiskit_model_losses],
        "qiskit_model_final_loss": qiskit_model_final_loss,
        "qiskit_model_final_accuracy": qiskit_model_final_accuracy,
        "merlin_model_accuracies": [float(v) for v in merlin_model_accuracies],
        "merlin_model_losses": [float(v) for v in merlin_model_losses],
        "merlin_model_final_loss": merlin_model_final_loss,
        "merlin_model_final_accuracy": merlin_model_final_accuracy,
    }

    json_str = json.dumps(json_payload, indent=4)
    current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"
    with open(current_dir + "bas_run_data.json", "w") as f:
        f.write(json_str)
    if generate_graph is True:
        plot_bas_run(
            accuracy_classical=classical_model_accuracies,
            accuracy_qiskit=qiskit_model_accuracies,
            accuracy_merlin=merlin_model_accuracies,
            loss_classical=classical_model_losses,
            loss_qiskit=qiskit_model_losses,
            loss_merlin=merlin_model_losses,
            run_dir=run_dir,
        )


# run_bas(num_epochs=20, generate_graph=True)
