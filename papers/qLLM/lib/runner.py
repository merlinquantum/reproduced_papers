import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _parse_config_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        if not all(isinstance(item, dict) for item in raw):
            raise ValueError("All items in config list must be dictionaries")
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"Unsupported config list type: {type(raw)}")
    config_string = raw.strip()
    if not config_string:
        return []

    try:
        config_list = json.loads(config_string)
        if not isinstance(config_list, list):
            raise ValueError("Config must be a list")
        if not all(isinstance(item, dict) for item in config_list):
            raise ValueError("All items in config list must be dictionaries")
        return config_list
    except json.JSONDecodeError:
        pass

    try:
        config_list = ast.literal_eval(config_string)
        if not isinstance(config_list, list):
            raise ValueError("Config must be a list")
        if not all(isinstance(item, dict) for item in config_list):
            raise ValueError("All items in config list must be dictionaries")
        return config_list
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid config format: {exc}") from exc


def _build_args(cfg: dict[str, Any]) -> SimpleNamespace:
    dataset_cfg = cfg.get("dataset", {})
    embeddings_cfg = cfg.get("embeddings", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})

    args = SimpleNamespace(
        dataset=dataset_cfg.get("name", "sst2"),
        eval_size=dataset_cfg.get("eval_size", 250),
        embeddings_dir=str(dataset_cfg.get("embeddings_dir", "embeddings")),
        model_name=embeddings_cfg.get(
            "model_name", "sentence-transformers/paraphrase-mpnet-base-v2"
        ),
        embedding_dim=model_cfg.get("embedding_dim", 768),
        hidden_dim=model_cfg.get("hidden_dim", 100),
        epochs=training_cfg.get("epochs", 5),
        learning_rate=training_cfg.get("learning_rate", 1e-5),
        batch_size=training_cfg.get("batch_size", 16),
        kernel_batch_size=training_cfg.get("kernel_batch_size", 32),
        model=model_cfg.get("name", "merlin-basic"),
        quantum_modes=model_cfg.get("quantum_modes", 8),
        no_bunching=model_cfg.get("no_bunching", False),
        photons=model_cfg.get("photons", 0),
        encoder_configs=_parse_config_list(model_cfg.get("encoder_configs")),
        pqc_config=_parse_config_list(model_cfg.get("pqc_config")),
        e_dim=model_cfg.get("e_dim", 1),
        input_state=model_cfg.get("input_state"),
        seed=cfg.get("seed", 42),
        device=cfg.get("device", "auto"),
        no_plot=runtime_cfg.get("no_plot", False),
        verbose=runtime_cfg.get("verbose", False),
    )
    return args


def setup_environment(args):
    """Set up random seeds and device"""
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.verbose:
        print(f"Using device: {device}")
        print("Configuration:")
        print(f"- Head training epochs: {args.epochs}")
        print(f"- Learning rate: {args.learning_rate}")

    return device


def train_model(model, train_dataset, eval_dataset, test_dataset, args):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        embeddings = torch.stack(
            [torch.as_tensor(item["embedding"], dtype=torch.float32) for item in batch]
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        return {"embedding": embeddings, "label": labels}

    # Prepare dataset
    train_dataset.set_format(type="torch", columns=["embedding", "label"])
    eval_dataset.set_format(type="torch", columns=["embedding", "label"])
    test_dataset.set_format(type="torch", columns=["embedding", "label"])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
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
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
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
        from .merlin_llm_models import QuantumClassifier

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
        from .merlin_llm_models import QuantumClassifierParallel

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
        from .merlin_llm_models import QuantumClassifierExpectation

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
        from .torchquantum_models import QLLM

        encoder_configs = args.encoder_configs or None
        qpu_config = args.pqc_config[0] if args.pqc_config else None
        model = QLLM(encoder_configs=encoder_configs, qpu_config=qpu_config)
        model_name = args.model
    elif args.model == "mlps":
        from .classical_models import MLPClassifier

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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    if args.model == "merlin-kernel":
        from .merlin_kernel import create_setfit_with_q_kernel

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
        accuracy_dict = {args.model: float(mean_accuracy)}

    elif args.model == "svm":
        # SVM with varying parameter counts (targeting 296 and 435 parameters)
        print("\nTraining SVM heads with varying parameter counts...")

        # Configuration 1: Target ~296 parameters (moderate regularization)
        print("   a. Training SVM targeting ~296 parameters...")

        model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
        model.fit(train_embeddings, train_dataset["label"])

        svc_296_val_preds = model.predict(eval_embeddings)
        svc_296_test_preds = model.predict(test_embeddings)
        svc_296_val_accuracy = accuracy_score(eval_dataset["label"], svc_296_val_preds)
        svc_296_test_accuracy = accuracy_score(
            test_dataset["label"], svc_296_test_preds
        )

        n_support_vectors_296 = int(model.n_support_.sum())

        print(
            f"   SVM (296 target) - Support vectors: {n_support_vectors_296}, Val: {svc_296_val_accuracy:.4f}, Test: {svc_296_test_accuracy:.4f}"
        )

        # Configuration 2: Target ~435 parameters (low regularization to use more support vectors)
        print("   b. Training SVM targeting ~435 parameters...")

        model = SVC(C=100.0, kernel="rbf", gamma="scale", probability=True)
        model.fit(train_embeddings, train_dataset["label"])

        svc_435_val_preds = model.predict(eval_embeddings)
        svc_435_test_preds = model.predict(test_embeddings)
        svc_435_val_accuracy = accuracy_score(eval_dataset["label"], svc_435_val_preds)
        svc_435_test_accuracy = accuracy_score(
            test_dataset["label"], svc_435_test_preds
        )

        n_support_vectors_435 = int(model.n_support_.sum())

        print(
            f"   SVM (435 target) - Support vectors: {n_support_vectors_435}, Val: {svc_435_val_accuracy:.4f}, Test: {svc_435_test_accuracy:.4f}"
        )

        accuracy_dict = {
            "svm_296": {
                "accuracy": float(svc_296_test_accuracy),
                "support_vectors": n_support_vectors_296,
            },
            "svm_435": {
                "accuracy": float(svc_435_test_accuracy),
                "support_vectors": n_support_vectors_435,
            },
        }

    elif args.model == "log-reg":
        print("\nTraining Logistic Regression head...")
        model = LogisticRegression()
        model.fit(train_embeddings, train_dataset["label"])

        lg_val_accuracy = accuracy_score(
            eval_dataset["label"], model.predict(eval_embeddings)
        )
        lg_test_accuracy = accuracy_score(
            test_dataset["label"], model.predict(test_embeddings)
        )

        print(
            f"Logistic Regression - Val: {lg_val_accuracy:.4f}, Test: {lg_test_accuracy:.4f}"
        )
        accuracy_dict = {args.model: lg_test_accuracy}

    return accuracy_dict


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    args = _build_args(cfg)

    def _skip(note: str) -> None:
        logger.warning(note)
        (run_dir / "SKIPPED.txt").write_text(f"{note}\n", encoding="utf-8")

    embeddings_dir = Path(args.embeddings_dir)
    required_files = [
        embeddings_dir / "train_embeddings.json",
        embeddings_dir / "eval_embeddings.json",
        embeddings_dir / "test_embeddings.json",
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        note = (
            "Embeddings are missing; generate them with "
            "`python utils/generate_embeddings.py` from `papers/qLLM/`.\n"
            f"Missing files:\n{missing_list}\n"
        )
        _skip(note.strip())
        return

    if args.model in {"svm", "log-reg"}:
        device = None
        np.random.seed(args.seed)
    else:
        try:
            device = setup_environment(args)
        except ImportError as exc:
            _skip(
                f"Missing dependency for model '{args.model}': {exc}. "
                "Install `papers/qLLM/requirements-full.txt` with a supported Python "
                "version (3.10–3.12) to run quantum models."
            )
            return
        if args.model.startswith("merlin") and args.photons == 0:
            args.photons = args.quantum_modes // 2

    logger.info("Loading embeddings from %s", embeddings_dir)
    from papers.shared.qLLM.data_utils import create_dataset_from_embeddings

    train_dataset = create_dataset_from_embeddings(str(embeddings_dir), "train")
    test_dataset = create_dataset_from_embeddings(str(embeddings_dir), "test")
    eval_dataset = create_dataset_from_embeddings(str(embeddings_dir), "eval")

    try:
        model_setup = pick_model(args, device)
    except ImportError as exc:
        note = (
            f"Missing dependency for model '{args.model}': {exc}. "
            "Install `papers/qLLM/requirements-full.txt` with a supported Python "
            "version (3.10–3.12) to run quantum models."
        )
        _skip(note)
        return
    model_name = list(model_setup.keys())[0]
    results: dict[str, Any] = {"model": model_name, "entries": []}

    if model_name == "kernel_method":
        logger.info("Training %s kernel method", args.model)
        accuracy_dict = train_kernel_method(
            args, train_dataset, eval_dataset, test_dataset
        )
        results["kernel_metrics"] = accuracy_dict
    else:
        logger.info("Training %s", model_name)
        for model in model_setup[model_name]:
            _, best_val_acc, test_acc = train_model(
                model, train_dataset, eval_dataset, test_dataset, args
            )
            entry = {
                "val_accuracy": best_val_acc,
                "test_accuracy": test_acc,
            }
            if model_name == "mlps":
                entry["model"] = repr(model)
            results["entries"].append(entry)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)
