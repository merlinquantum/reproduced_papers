import argparse
import ast
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classical_utils import MLPClassifier, evaluate
from data_utils import create_dataset_from_embeddings
from merlin_kernel import create_setfit_with_q_kernel

# Import utility modules
from merlin_llm_utils import (
    QuantumClassifier,
    QuantumClassifierExpectation,
    QuantumClassifierParallel,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchquantum_utils import QLLM


def parse_config_list(config_string: str) -> list[dict[str, Any]]:
    """
    Parse a string representation of list[dict] into actual Python objects.
    Supports both JSON and Python literal formats.
    """
    config_string = config_string.strip()

    # Try JSON parsing first (more robust)
    try:
        config_list = json.loads(config_string)
        if not isinstance(config_list, list):
            raise ValueError("Config must be a list")
        for item in config_list:
            if not isinstance(item, dict):
                raise ValueError("All items in config list must be dictionaries")
        return config_list
    except json.JSONDecodeError:
        pass

    # Fall back to Python literal evaluation
    try:
        config_list = ast.literal_eval(config_string)
        if not isinstance(config_list, list):
            raise ValueError("Config must be a list")
        for item in config_list:
            if not isinstance(item, dict):
                raise ValueError("All items in config list must be dictionaries")
        return config_list
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid config format: {e}") from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train QLLM with classical and quantum heads"
    )

    ### Dataset parameters ###
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset name")
    parser.add_argument(
        "--eval-size", type=int, default=250, help="Validation set size"
    )

    ### Model parameters ###
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=768, help="Embedding dimension"
    )

    ### embedding compression ###
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=100,
        help="Hidden dimension for MLP/Quantum layers",
    )

    ### MODEL HEAD TRAINING ###
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for classification head training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for body fine-tuning",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=32,
        help="Batch size for kernel computation in merlin-kernel method",
    )

    ### MODEL CHOICE ###
    parser.add_argument(
        "--model",
        choices=[
            "merlin-basic",
            "merlin-parallel",
            "merlin-expectation",
            "merlin-kernel",
            "torchquantum",
            "mlps",
            "svm",
            "log-reg",
        ],
        default="merlin-basic",
    )

    ### MerLin parameters ###
    parser.add_argument(
        "--quantum-modes",
        type=int,
        default=8,
        help="Number of quantum modes for MerLin model",
    )
    parser.add_argument(
        "--no-bunching",
        action="store_true",
        help="No bunching parameter for the MerLin model",
    )
    parser.add_argument(
        "--photons",
        type=int,
        default=0,
        help="Number of photons max (0 stands for modes/2) to be used in the interferometer",
    )

    ### TorchQuantum parameters ###
    parser.add_argument(
        "--encoder-configs",
        type=parse_config_list,
        default=[
            {"n_qubits": 10, "n_layers": 1, "connectivity": 1},
            {"n_qubits": 10, "n_layers": 1, "connectivity": 1},
        ],
        help="List of config dictionaries as JSON or Python literal",
    )

    parser.add_argument(
        "--pqc-config",
        type=parse_config_list,
        default=[
            {"n_qubits": 10, "n_main_layers": 2, "connectivity": 1, "n_reuploading": 2}
        ],
        help="List of config dictionaries as JSON or Python literal",
    )

    # Both models: parallel encoding #
    parser.add_argument(
        "--e_dim",
        type=int,
        default=1,
        help="Parallel encoder",
    )

    # Execution parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def collate_fn(batch):
    embeddings = torch.stack(
        [torch.tensor(item["embedding"], dtype=torch.float32) for item in batch]
    )
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {"embedding": embeddings, "label": labels}


def setup_environment(args):
    """Set up random seeds and device"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.verbose:
        print(f"Using device: {device}")
        print("Configuration:")
        print(f"- Samples per class: {args.samples_per_class}")
        print(f"- Head training epochs: {args.epochs}")
        print(f"- Learning rate: {args.learning_rate}")

    return device


def train_model(model, train_dataset, eval_dataset, test_dataset, args):
    # Prepare dataset
    train_dataset.set_format(type="torch", columns=["embedding", "label"])
    eval_dataset.set_format(type="torch", columns=["embedding", "label"])
    test_dataset.set_format(type="torch", columns=["embedding", "label"])

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training variables
    best_val_acc = 0.0
    best_model_state = None

    print(f"Starting training for {args.epochs} epochs...")
    print("-" * 50)

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch_x = batch["embedding"]
            batch_y = batch["label"]
            optimizer.zero_grad()
            # print(f"Batch x of shape {batch_x.shape}")
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch["embedding"]
                batch_y = batch["label"]
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
            print(
                f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
            )
            print(
                f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%"
            )
            print(f"Best Val Acc: {best_val_acc:.2f}%")
            print("-" * 30)

    # Load best model and evaluate on test set
    print("\nTraining completed!")
    print(f"Loading best model (Val Acc: {best_val_acc:.2f}%)")
    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch["embedding"]
            batch_y = batch["label"]
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("=" * 50)

    return model, best_val_acc, test_acc


def pick_model(args, device):
    if args.model == "merlin-basic":
        print(
            f"Embedding dimension: {args.embedding_dim}, hidden_dim : {args.hidden_dim}"
        )
        model = QuantumClassifier(
            input_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            modes=args.quantum_modes,
            input_state=args.input_state,
            num_classes=2,
            device=device,
            no_bunching=args.no_bunching,
        )
        model_name = args.model
    elif args.model == "merlin-parallel":
        model = QuantumClassifierParallel(
            input_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            modes=args.quantum_modes,
            input_state=args.input_state,
            device=device,
            e=args.e_dim,
        )
        model_name = args.model
    elif args.model == "merlin-expectation":
        model = QuantumClassifierExpectation(
            input_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            modes=args.quantum_modes,
            input_state=args.input_state,
            device=device,
            e=args.e_dim,
        )
        model_name = args.model
    elif args.model == "torchquantum":
        model = QLLM(
            encoder_configs=args.encoder_configs, qpu_config=args.pqc_config[0]
        )
        model_name = args.model
    elif args.model == "mlps":
        model = []
        hidden_dims = [0, 48, 96, 144, 192]
        for hidden_dim in hidden_dims:
            mlp = MLPClassifier(input_dim=args.embedding_dim, hidden_dim=hidden_dim)
            model.append(mlp)
        model_name = args.model
    else:
        model = None
        model_name = "kernel_method"

    return {model_name: [model] if not args.model == "mlps" else model}


def train_kernel_method(args, train_dataset, eval_dataset, test_dataset):
    train_embeddings = np.array(train_dataset["embedding"])
    eval_embeddings = np.array(eval_dataset["embedding"])
    test_embeddings = np.array(test_dataset["embedding"])
    if args.model == "merlin-kernel":
        print("Training a quantum Kernel with MerLin")
        model, kernel = create_setfit_with_q_kernel(
            modes=args.quantum_modes, photons=args.photons
        )
        print("\n -> Computing K_train")
        k_train = kernel(train_embeddings)
        print(f"... Done (K_train of shape {k_train.shape}) !")
        print("\n -> Fitting model to K_train")
        model.fit(k_train, train_dataset["label"])

        print("\n -> Computing K_test with batched evaluation")
        print(
            f"Type of eval embeddings: {type(eval_embeddings)} and train embeddings: {type(train_embeddings)}"
        )

        # Batched evaluation of test set
        batch_size = (
            args.kernel_batch_size if hasattr(args, "kernel_batch_size") else 32
        )
        n_test_samples = len(test_embeddings)
        print(f"Evaluating {n_test_samples} test samples in batches of {batch_size}")

        # Initialize list to store kernel values for each batch
        accuracies = []
        for i in range(0, n_test_samples, batch_size):
            end_idx = min(i + batch_size, n_test_samples)
            batch_embeddings = test_embeddings[i:end_idx]
            batch_labels = test_dataset["label"][i:end_idx]

            # Compute kernel for this batch
            k_batch = kernel(batch_embeddings, train_embeddings)
            y_pred_batch = model.predict(k_batch)
            accuracy = accuracy_score(batch_labels, y_pred_batch)
            print(
                f"Processing batch {i // batch_size + 1}/{(n_test_samples + batch_size - 1) // batch_size} (samples {i + 1}-{end_idx}) - accuracy: {accuracy * 100:.2f}%"
            )
            accuracies.append(accuracy)
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f"Mean Accuracy: {mean_accuracy * 100:.4f}")
        accuracy_dict = {args.model: accuracy}

    elif args.model == "svm":
        # SVM with varying parameter counts (targeting 296 and 435 parameters)
        if args.verbose:
            print("\nTraining SVM heads with varying parameter counts...")

        # Configuration 1: Target ~296 parameters (moderate regularization)
        if args.verbose:
            print("   a. Training SVM targeting ~296 parameters...")

        model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
        model.fit(train_embeddings, train_dataset["label"])

        svc_296_val_accuracy, _ = evaluate(
            model, eval_embeddings, eval_dataset["label"]
        )
        svc_296_test_accuracy, _ = evaluate(
            model, test_embeddings, test_dataset["label"]
        )

        n_support_vectors_296 = model.n_support_.sum()

        if args.verbose:
            print(
                f"   SVM (296 target) - Support vectors: {n_support_vectors_296}, Val: {svc_296_val_accuracy:.4f}, Test: {svc_296_test_accuracy:.4f}"
            )

        # Configuration 2: Target ~435 parameters (low regularization to use more support vectors)
        if args.verbose:
            print("   b. Training SVM targeting ~435 parameters...")

        model = SVC(C=100.0, kernel="rbf", gamma="scale", probability=True)
        model.fit(train_embeddings, train_dataset["label"])

        svc_435_val_accuracy, _ = evaluate(
            model, eval_embeddings, eval_dataset["label"]
        )
        svc_435_test_accuracy, _ = evaluate(
            model, test_embeddings, test_dataset["label"]
        )

        n_support_vectors_435 = model.n_support_.sum()

        if args.verbose:
            print(
                f"   SVM (435 target) - Support vectors: {n_support_vectors_435}, Val: {svc_435_val_accuracy:.4f}, Test: {svc_435_test_accuracy:.4f}"
            )

        accuracy_dict = {
            "svm_296": {
                "accuracy": svc_296_test_accuracy,
                "suppport_vectors": n_support_vectors_296,
            },
            "svm_435": {
                "accuracy": svc_435_test_accuracy,
                "support_vectors": n_support_vectors_435,
            },
        }

    elif args.model == "log-reg":
        if args.verbose:
            print("\nTraining Logistic Regression head...")
        model = LogisticRegression()
        model.fit(train_embeddings, train_dataset["label"])

        lg_val_accuracy, _ = evaluate(model, eval_embeddings, eval_dataset["label"])
        lg_test_accuracy, _ = evaluate(model, test_embeddings, test_dataset["label"])

        if args.verbose:
            print(
                f"Logistic Regression - Val: {lg_val_accuracy:.4f}, Test: {lg_test_accuracy:.4f}"
            )
        accuracy_dict = {args.model: lg_test_accuracy}

    return accuracy_dict


if __name__ == "__main__":
    print("Entering main loop")
    args = parse_args()
    args.input_state = None
    print("Parsing done")
    device = setup_environment(args)
    print("Device selected")
    use_normalization = False

    if args.model[:6] == "merlin":
        if args.photons == 0:
            args.photons = args.quantum_modes // 2

    # load data
    print("Loading pre-computed embeddings from ./embeddings directory...")
    train_dataset = create_dataset_from_embeddings("./embeddings", split_name="train")
    test_dataset = create_dataset_from_embeddings("./embeddings", split_name="test")
    eval_dataset = create_dataset_from_embeddings("./embeddings", split_name="eval")

    model_setup = pick_model(args, device)
    model_name = list(model_setup.keys())[0]

    if model_name == "kernel_method":
        print(f"\n{'=' * 60}")
        print(f"\n  Training {args.model} method")
        print(f"\n{'=' * 60}")
        accuracy_dict = train_kernel_method(
            args, train_dataset, eval_dataset, test_dataset
        )

    else:
        print(f"\n{'=' * 60}")
        print(f"\n  Training the {model_name}")
        print(f"\n{'=' * 60}")
        for model in model_setup[model_name]:
            trained_model, best_val_acc, test_acc = train_model(
                model, train_dataset, eval_dataset, test_dataset, args
            )

            print(f"\n{'*' * 60}")
            if model_name == "mlps":
                print(model)
            print(
                f"\n  Best model ({model_name}) with a TEST accuracy of {test_acc:.4f} and a VAL accuracy of {best_val_acc:.4f} "
            )
            print(f"\n{'*' * 60}")

    print(f"\n{'=' * 60}")
    print("\n  Training is complete")
    print(f"\n{'=' * 60}")
