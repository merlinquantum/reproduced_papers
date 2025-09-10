"""
Classical model training utilities for qLLM experiments.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# the `ModelWrapper` class provides a unified interface for
## handling tokenization
## forward passes with sentence transformer models
# here, we can work with different model architectures seamlessly


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def tokenize(self, texts):
        """
        Delegates tokenization to the underlying model.

        Args:
            texts (List[str]): List of text strings to tokenize

        Returns:
            Dict or Tensor: Tokenized inputs in the format expected by the model
        """
        try:
            # Try to use the tokenize method of the underlying model
            return self.model.tokenize(texts)
        except AttributeError:
            # If the model doesn't have a tokenize method, try alternative approaches
            if hasattr(self.model, "tokenizer"):
                return self.model.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
            elif hasattr(self.model, "_first_module") and hasattr(
                self.model._first_module, "tokenizer"
            ):
                return self.model._first_module.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
            else:
                raise ValueError(
                    "Unable to tokenize texts with this model. Please provide a model that has a tokenize or tokenizer method."
                ) from None

    def forward(self, inputs):
        """
        Process inputs through the model to get embeddings.

        Args:
            inputs: Can be raw text strings or pre-tokenized inputs

        Returns:
            torch.Tensor: The sentence embeddings
        """
        try:
            # Handle different input formats
            if isinstance(inputs, dict) and all(
                isinstance(v, torch.Tensor) for v in inputs.values()
            ):
                outputs = self.model(inputs)
            elif isinstance(inputs, list) and all(isinstance(t, str) for t in inputs):
                tokenized = self.tokenize(inputs)
                device = next(self.model.parameters()).device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                outputs = self.model(tokenized)
            else:
                outputs = self.model(inputs)

            # Extract embeddings from various output formats
            if isinstance(outputs, dict) and "sentence_embedding" in outputs:
                return outputs["sentence_embedding"]
            elif isinstance(outputs, dict) and "pooler_output" in outputs:
                return outputs["pooler_output"]
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0]
            else:
                return outputs
        except Exception as e:
            raise ValueError(f"Error during forward pass: {str(e)}") from e


## evaluation function ##


def evaluate(model, embeddings, labels):
    """
    Evaluate SetFit model on given texts and labels.

    Args:
        model: model to evaluate
        texts: List of text strings to classify
        labels: True labels for evaluation

    Returns:
        tuple: (accuracy, predictions)
    """
    batch_size = 16
    num_samples = embeddings.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_embeddings = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # batch_texts = texts[start_idx:end_idx]

            # Get embeddings
            """batch_embeddings = model.model_body.encode(
                batch_texts, convert_to_tensor=True
            )"""
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_embeddings_cpu = batch_embeddings  # .detach().cpu().numpy()

            all_embeddings.extend(batch_embeddings_cpu)

    # Use the classification head to predict
    predictions = model.predict(np.array(all_embeddings))
    accuracy = accuracy_score(labels, predictions)
    return accuracy, predictions


# MLPClassifier and its wrapper#


class MLPClassifier(nn.Module):
    """3-layer MLP classifier with dropout regularization"""

    def __init__(self, input_dim, hidden_dim=100, num_classes=2):
        super().__init__()
        self.layers = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
            if hidden_dim > 0
            else nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for the MLP classifier"""

    def __init__(
        self,
        input_dim=768,
        hidden_dim=100,
        num_classes=2,
        lr=0.001,
        gamma=0.99,
        wd=1e-6,
        epochs=100,
        batch_size=32,
        device=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.classes_ = None
        self.train_accuracies = []
        self.val_accuracies = []

    def fit(self, x, y, val_x=None, val_y=None):
        """Train the MLP classifier with optional validation data"""
        # Store unique classes
        self.classes_ = np.unique(y)

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        if val_x is not None and val_y is not None:
            val_x = torch.tensor(val_x, dtype=torch.float32).to(self.device)
            val_y = torch.tensor(val_y, dtype=torch.long).to(self.device)

        # Initialize the model
        self.model = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_),
        ).to(self.device)

        print(
            f"Number of parameters in MLP head: {sum([p.numel() for p in self.model.parameters()])}"
        )

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        print(f"\n --- \n Training MLP head with lr = {self.lr}, weight_decay = ")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Add learning rate scheduler based on validation loss
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        best_val = 0
        self.best_model_state_dict = None
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            indices = torch.randperm(len(x))
            total_loss = 0
            correct_train = 0
            total_train = 0

            for i in range(0, len(x), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch_x = x[batch_indices]
                batch_y = y_tensor[batch_indices]

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            train_accuracy = 100 * correct_train / total_train
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            val_accuracy = 0
            # val_loss = 0
            if val_x is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_x)
                    # val_loss = criterion(val_outputs, val_y).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (
                        100 * (val_predicted == val_y).sum().item() / val_y.size(0)
                    )
                    self.val_accuracies.append(val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

            # Step the scheduler with validation loss (if validation data is provided)
            if val_x is not None and val_y is not None:
                scheduler.step()

            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(x) // self.batch_size + 1)
                if val_x is not None and val_y is not None:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}% (Best Val: {best_val:.2f}%)  [lr = {scheduler.get_last_lr()[0]}]"
                    )
                else:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
                    )
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print(f"Best model loaded with validation accuracy: {best_val:.2f}%")
        self.best_val = best_val
        return self

    def predict(self, x):
        """Predict classes for samples"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes_[predicted.cpu().numpy()]

    def predict_proba(self, x):
        """Predict class probabilities"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            return probabilities


def replace_setfit_head_with_mlp(
    model,
    input_dim=768,
    hidden_dim=100,
    num_classes=2,
    epochs=100,
    lr=0.001,
    gamma=0.99,
    wd=1e-6,
):
    """Replace the classification head of a SetFitModel with an MLP."""
    # Get the device the model is on
    device = next(model.model_body.parameters()).device

    # Create new MLP head
    mlp_head = MLPClassifierWrapper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        wd=wd,
        device=device,
    )

    # Replace the model head
    model.model_head = mlp_head

    return model


def generate_embeddings(sentence_transformer, train_dataset, args):
    """Generate embeddings from the fine-tuned model"""
    if args.verbose:
        print("\nGenerating embeddings for training data...")

    sentence_transformer.eval()
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        num_batches = (
            len(train_dataset["sentence"]) + args.batch_size - 1
        ) // args.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Encoding"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(train_dataset["sentence"]))

            batch_texts = train_dataset["sentence"][start_idx:end_idx]
            batch_labels = train_dataset["label"][start_idx:end_idx]

            batch_embeddings = sentence_transformer.encode(
                batch_texts, convert_to_tensor=True
            )
            batch_embeddings_cpu = batch_embeddings.detach().cpu().numpy()

            for emb, lbl in zip(batch_embeddings_cpu, batch_labels):
                train_embeddings.append(emb)
                train_labels.append(lbl)

    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)

    if args.verbose:
        print(f"Embeddings shape: {train_embeddings.shape}")
        print(f"Labels shape: {train_labels.shape}")
    # scaling on the embeddings
    global_train_max = train_embeddings.max()
    global_train_min = train_embeddings.min()
    # min to 0 and max to 1
    train_embeddings_scaled = (train_embeddings - global_train_min) / (
        global_train_max - global_train_min
    )
    return (
        train_embeddings,
        train_embeddings_scaled,
        train_labels,
        global_train_max,
        global_train_min,
    )


def generate_val_test_emnbeddings(
    sentence_transformer,
    eval_dataset,
    test_dataset,
    normalization,
    global_train_max,
    global_train_min,
):
    eval_embeddings = sentence_transformer.encode(eval_dataset["sentence"])

    # eval embeddings
    if normalization:
        eval_embeddings = (eval_embeddings - global_train_min) / (
            global_train_max - global_train_min
        )
        eval_embeddings = torch.tensor(np.clip(eval_embeddings, 0, 1))
    else:
        eval_embeddings = torch.tensor(eval_embeddings)

    # test embeddings
    test_embeddings = sentence_transformer.encode(test_dataset["sentence"])
    if normalization:
        test_embeddings = (test_embeddings - global_train_min) / (
            global_train_max - global_train_min
        )
        test_embeddings = torch.tensor(np.clip(test_embeddings, 0, 1))
    else:
        test_embeddings = torch.tensor(test_embeddings)

    return eval_embeddings, test_embeddings
