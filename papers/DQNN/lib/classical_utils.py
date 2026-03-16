"""
Classical CNN utilities for DQNN experiments.

This module provides models, pruning helpers, and dataset/training utilities
used alongside the photonic quantum train.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class SharedWeightFC(nn.Module):
    """
    Fully connected layer with repeated shared weights.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    shared_rows : int
        Number of shared rows to repeat to form the weight matrix.
    """

    def __init__(self, in_features: int, out_features: int, shared_rows: int):
        """
        Initialize the shared-weight fully connected layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        shared_rows : int
            Number of shared rows to repeat to form the weight matrix.
        """
        super().__init__()
        self.shared_weights = nn.Parameter(torch.randn(shared_rows, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.out_features = out_features
        self.shared_rows = shared_rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the shared-weight projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_features).
        """
        weight_matrix = self.shared_weights.repeat(
            self.out_features // self.shared_rows, 1
        )
        return torch.matmul(x, weight_matrix.t()) + self.bias


class CNNModel(nn.Module):
    """
    Simple CNN for MNIST with optional weight sharing.

    Parameters
    ----------
    use_weight_sharing : bool, optional
        Whether to use shared weights in the first FC layer. Default is False.
    shared_rows : int, optional
        Number of shared rows when using weight sharing. Default is 10.
    """

    def __init__(self, use_weight_sharing: bool = False, shared_rows: int = 10):
        """
        Initialize the CNN model.

        Parameters
        ----------
        use_weight_sharing : bool, optional
            Whether to use shared weights in the first FC layer. Default is False.
        shared_rows : int, optional
            Number of shared rows when using weight sharing. Default is 10.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=5)

        if use_weight_sharing:
            self.fc1 = SharedWeightFC(
                in_features=12 * 4 * 4, out_features=20, shared_rows=shared_rows
            )
            self.fc2 = nn.Linear(20, 10)
        else:
            self.fc1 = nn.Linear(12 * 4 * 4, 20)
            self.fc2 = nn.Linear(20, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Model logits of shape (batch, 10).
        """
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def apply_pruning(model: nn.Module, amount: float = 0.3):
    """
    Apply structured pruning to convolutional and fully connected layers.

    Parameters
    ----------
    model : torch.nn.Module
        Model to prune in-place.
    amount : float, optional
        Fraction of filters/neurons to prune. Default is 0.3.

    Returns
    -------
    None
    """
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
            print(
                f"Applied structured pruning to Conv2d layer {name} with {amount * 100:.1f}% filters pruned."
            )
        elif isinstance(layer, nn.Linear):
            prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=1)
            print(
                f"Applied structured pruning to Linear layer {name} with {amount * 100:.1f}% neurons pruned."
            )


def remove_pruning(model: nn.Module):
    """
    Remove pruning masks to finalize the reduced model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with pruning re-parameterizations.

    Returns
    -------
    None
    """
    for _name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(layer, "weight")
            except ValueError:
                pass  # Layer was not pruned


class MaskedAdam(torch.optim.Adam):
    """
    Adam optimizer that zeros gradients for masked parameters.

    Parameters
    ----------
    params : iterable
        Parameters to optimize.
    **kwargs
        Additional Adam optimizer keyword arguments.
    """

    def __init__(self, params: list[float], **kwargs):
        """
        Initialize the masked Adam optimizer.

        Parameters
        ----------
        params : List[float]
            Parameters to optimize.
        **kwargs
            Additional Adam optimizer keyword arguments.
        """
        super().__init__(params, **kwargs)

    def step(self, closure: callable = None):
        """
        Perform a single optimization step with masking.

        Parameters
        ----------
        closure : callable, optional
            Optional closure that reevaluates the model and returns the loss.

        Returns
        -------
        None
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad_mask = param.data != 0
                param.grad.data.mul_(grad_mask)
        super().step(closure)


def train_classical_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    use_pruning: bool = False,
    pruning_amount: float = 0.5,
    use_weight_sharing: bool = False,
    shared_rows: int = 10,
) -> CNNModel:
    """
    Train a classical CNN baseline and evaluate on validation data.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    num_epochs : int
        Number of training epochs.
    use_pruning : bool, optional
        Whether to apply structured pruning. Default is False.
    pruning_amount : float, optional
        Fraction of weights to prune if pruning is enabled. Default is 0.5.
    use_weight_sharing : bool, optional
        Whether to use shared weights in the first FC layer. Default is False.
    shared_rows : int, optional
        Number of shared rows when using weight sharing. Default is 10.

    Returns
    -------
    CNNModel
        Trained CNN model.
    """
    learning_rate = 1e-3

    model = CNNModel(use_weight_sharing=use_weight_sharing, shared_rows=shared_rows)

    if use_pruning:
        apply_pruning(model, amount=pruning_amount)
        remove_pruning(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if use_pruning:
        optimizer = MaskedAdam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_classical_parameter = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("# of parameters in classical CNN model: ", num_classical_parameter)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

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
    return model
