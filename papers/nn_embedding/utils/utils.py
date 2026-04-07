import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import merlin as ml
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import gamma as gamma_fun
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from papers.nn_embedding.utils.merlin_model_utils import assign_params

############################################################################################################
# From the code of the paper


def create_random_pairs(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return (
        torch.stack(X1_new),
        torch.stack(X2_new),
        torch.as_tensor(Y_new, dtype=torch.float32),
    )


def pick_random_data(batch_size, X, Y):
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = torch.stack([X[i] for i in batch_index])
    Y_batch = torch.as_tensor([Y[i] for i in batch_index], dtype=torch.long)
    return X_batch, Y_batch


def calculate_distance(
    rho0: torch.Tensor, rho1: torch.Tensor, distance: str = "Trace"
) -> float:
    rho_diff = rho1 - rho0
    if distance == "Trace":
        eigvals = torch.linalg.eigvals(rho_diff)
        return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
    elif distance == "Hilbert-Schmidt":
        return 0.5 * torch.trace(rho_diff @ rho_diff)
    else:
        raise ValueError("No distance with that name")


def loss_lower_bound(rhos_0: torch.Tensor, rhos_1: torch.Tensor) -> float:
    """
    Empirical risk is lower bounded by this quantity
    """
    N = rhos_0.size(dim=0) + rhos_1.size(dim=1)

    return 0.5 - calculate_distance(
        torch.sum(rhos_0, dim=0) / N, torch.sum(rhos_1, dim=0) / N
    )


def get_error_bound(weights: np.ndarray, Kernel: np.ndarray, Y_train: np.ndarray):
    N = len(Y_train)
    # TODO Add it to compute_kernel_instead!
    # Per copilot to make sure that even numerically, the kernel matrix is trace preserving
    # The fidelity kernel is theoretically PSD, but float32 quantum simulation
    # introduces small negative eigenvalues. Project onto the PSD cone.
    K = Kernel.astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0)
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T

    Y = Y_train.astype(np.float64)
    error_list = []

    for weight in weights:
        Kernel_MP = np.linalg.pinv(K + weight * np.eye(N), hermitian=True)
        val = Y @ Kernel_MP @ K @ Kernel_MP @ Y.transpose() / N
        error_list.append(np.sqrt(val))

    return np.array(error_list)


def random_unitary_gate_based(n):
    """
    Return a Haar distributed random unitary from U(N)
    """

    Z = np.random.randn(2**n, 2**n) + 1.0j * np.random.randn(2**n, 2**n)
    [Q, R] = np.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)


def haar_integral_gate_based(num_qubits, samples):
    """
    Return calculation of Haar Integral for a specified number of samples.
    """

    n = num_qubits
    randunit_density = np.zeros((4**n, 4**n), dtype=complex)

    zero_state = np.zeros(4**n, dtype=complex)
    zero_state[0] = 1

    for _ in range(samples):
        U = random_unitary_gate_based(n)
        U = np.kron(U, U)
        A = np.matmul(zero_state, U).reshape(-1, 1)
        randunit_density += np.kron(A, A.conj().T)

    randunit_density /= samples

    return randunit_density


def random_state_photonics(dim: int):
    """
    Return a Haar distributed random state vector of dimension dim.
    """
    z = np.random.randn(dim) + 1.0j * np.random.randn(dim)
    return state_vector_to_density_matrix(z / np.linalg.norm(z))


def haar_integral_photonics(dim: int, samples):
    """
    Return calculation of Haar Integral for a specified number of samples.
    Dim is the number of possible states (ex: m choose n for unbunched)
    """

    randunit_density = np.zeros((dim**2, dim**2), dtype=complex)

    for _ in range(samples):
        A = random_state_photonics(dim)
        randunit_density += np.kron(A, A)

    randunit_density /= samples

    return randunit_density


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def two_design_deviation_gate_based(rhos: torch.Tensor, num_qubits: int, N: int):
    """
    N is the number of samples used to define the rhos
    """
    N = rhos.size(dim=0)
    rhos = kron(rhos, rhos)
    rho = torch.sum(rhos, dim=0) / N
    rho = rho.detach().numpy()
    exp = np.linalg.norm(rho - haar_integral_gate_based(num_qubits, N))
    return exp**2


def two_design_deviation_photonics(rhos: torch.Tensor, dim: int, N: int):
    """
    N is the number of samples used to define the rhos
    Dim is the number of possible states (ex: m choose n for unbunched)
    """
    N = rhos.size(0)
    rhos = kron(rhos, rhos)
    rho = torch.sum(rhos, 0) / N
    rho = rho.detach().numpy()
    exp = np.linalg.norm(rho - haar_integral_photonics(dim, N))
    return exp**2


def kernel_variance(kernel_matrix: torch.Tensor) -> float:
    N = kernel_matrix.size(0)
    Kernel_offD = []
    for i in range(N):
        for j in range(i + 1, N):
            Kernel_offD.append(kernel_matrix[i][j])

    Kernel_offD = np.array(Kernel_offD)
    return Kernel_offD.std() ** 2


############################################################################################################


class LinearLoss(nn.Module):
    def forward(self, labels, predictions):
        labels = labels.to(predictions.dtype)
        # Same as 1-(l * p) where the labels are -1 and 1
        return (labels + predictions - (2 * labels * predictions)).mean()


def state_vector_to_density_matrix(
    x: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.ndim == 1:
            return torch.outer(x, torch.conj(x))
        if x.ndim == 2:
            return x.unsqueeze(-1) * torch.conj(x).unsqueeze(-2)
        raise ValueError("x must have shape (state_dim,) or (batch_size, state_dim)")
    else:
        return np.outer(x, x.conj())


def create_basic_gate_based_model(
    num_qubits: int,
    quantum_embedding_circuit: callable,
    quantum_classifier_circuit: callable,
    quantum_classifier_params_shape: tuple[int, ...],
    num_classes: int = 2,
):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface="torch")
    def complete_circuit(inputs, classifier_params):
        quantum_embedding_circuit(inputs)
        quantum_classifier_circuit(classifier_params)
        return qml.probs(wires=range(num_qubits))

    complete_circuit_layer = qml.qnn.TorchLayer(
        complete_circuit,
        weight_shapes={"classifier_params": quantum_classifier_params_shape},
    )

    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(2**num_qubits, num_classes)
            self.complete_circuit_layer = complete_circuit_layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probs = torch.stack(
                tuple(self.complete_circuit_layer(sample) for sample in x),
                dim=0,
            )
            return self.output_grouper(probs)

    return BasicModel()


def create_trainable_embedding_gate_based_model(
    num_qubits: int,
    quantum_embedding_circuit: callable,
    quantum_classifier_circuit: callable,
    embedding_params_shape: tuple[int, ...],
    quantum_classifier_params_shape: tuple[int, ...],
    num_classes: int = 2,
):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface="torch")
    def complete_circuit(inputs, embedding_params, classifier_params):
        quantum_embedding_circuit(inputs, embedding_params)
        quantum_classifier_circuit(classifier_params)
        return qml.probs(wires=range(num_qubits))

    complete_circuit_layer = qml.qnn.TorchLayer(
        complete_circuit,
        weight_shapes={
            "embedding_params": embedding_params_shape,
            "classifier_params": quantum_classifier_params_shape,
        },
    )

    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(2**num_qubits, num_classes)
            self.complete_circuit_layer = complete_circuit_layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probs = torch.stack(
                tuple(self.complete_circuit_layer(sample) for sample in x),
                dim=0,
            )
            return self.output_grouper(probs)

    return BasicModel()


def create_basic_merlin_model(
    quantum_embedding_layer: ml.QuantumLayer,
    quantum_classifier: ml.QuantumLayer,
    num_classes: int = 2,
):
    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(
                quantum_classifier.output_size, num_classes
            )
            self.embedder = deepcopy(quantum_embedding_layer)
            for param in self.embedder.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                x = x.reshape(x.size(0), -1)
                states = assign_params(self.embedder, x)

            probs = quantum_classifier(states)

            return self.output_grouper(probs)

    return BasicModel()


def create_trainable_embedding_merlin_model(
    quantum_embedding_layer: ml.QuantumLayer,
    quantum_classifier: ml.QuantumLayer,
    num_classes: int = 2,
):
    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_grouper = ml.LexGrouping(
                quantum_classifier.output_size, num_classes
            )
            self.embedder = deepcopy(quantum_embedding_layer)
            for param in self.embedder.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            states = self.embedder(x)

            probs = quantum_classifier(states)

            return self.output_grouper(probs)

    return BasicModel()


def to_serializable_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, np.generic):
        return values.item()
    if isinstance(values, dict):
        return {key: to_serializable_list(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [to_serializable_list(value) for value in values]
    return values


def randomize_trainable_parameters(module: nn.Module) -> None:
    """Force a fresh random initialization for each repetition.

    Classical PyTorch modules keep their native ``reset_parameters`` behavior.
    Their freshly initialized values are then lightly reshuffled so repetitions
    stay independent while preserving the original value range. Only Merlin
    quantum-layer trainable tensors are additionally resampled in ``[-pi, pi]``.
    """
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()
            if not isinstance(submodule, ml.QuantumLayer):
                for param in submodule.parameters(recurse=False):
                    if param.requires_grad and param.numel() > 1:
                        with torch.no_grad():
                            shuffled = param.reshape(-1)[
                                torch.randperm(param.numel(), device=param.device)
                            ].reshape_as(param)
                            param.copy_(shuffled)

    if isinstance(module, ml.QuantumLayer):
        for param in module.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    param.uniform_(-torch.pi, torch.pi)


class TransparentModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def get_local_dimension(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    num_samples: int = 100,
    fim_subsample: int = 500,
) -> float:
    d = sum(param.numel() for param in model.parameters())
    eye = np.eye(d)
    params = [p for p in model.parameters()]
    total_size = y.size(0)

    values = []

    for n in range(250, total_size + 1, (total_size - 250) // 5):
        gamma = 1
        kappa = gamma * n / (2 * np.pi * np.log(n))
        params_to_sample = create_param_ensemble(
            params, d, epsilon=epsilon, num_samples=num_samples
        )

        # COPILOT, LOI DES GRANDS NOMBRES: Subsample data for FIM — it is a statistical expectation so a
        # representative subset is sufficient and avoids O(n) forward passes.
        fim_n = min(fim_subsample, n)
        indices = np.random.choice(n, fim_n, replace=False)
        subset = TensorDataset(x[indices], y[indices])
        sub_loader = DataLoader(subset, shuffle=True)

        trace_sum = 0
        fims = []
        for point in params_to_sample:
            # Assign the params
            model.eval()
            with torch.no_grad():
                for value, param in zip(point, model.parameters()):
                    param.copy_(value)
            model.train()

            # Compute the empirical Fisher information matrix
            fim = torch.zeros(d, d)
            count = 0
            for x_batch, _ in sub_loader:
                output = model(x_batch)
                log_output = torch.log(output + 1e-10)
                for k in range(output.shape[-1]):
                    model.zero_grad()
                    log_output[0, k].backward(retain_graph=True)
                    grads = torch.cat(
                        [
                            p.grad.flatten()
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                    fim += torch.outer(grads, grads) * output[0, k].detach()
                count += 1
            fim = fim.detach().numpy() / max(count, 1)
            fims.append(fim)
            trace_sum += np.trace(fim)

        fim_normalisation_factor = d * num_samples * kappa / trace_sum

        # Compute the integral using slogdet (numerically stable for large d)
        integral = 0
        for fim in fims:
            _, logdet = np.linalg.slogdet(eye + fim_normalisation_factor * fim)
            integral += np.exp(0.5 * logdet)

        integral /= num_samples

        values.append((2 * np.log(integral)) / (np.log(kappa)))
        print(f"Led of {values[-1]}")

    return values


def create_param_ensemble(
    params: list[torch.Tensor], d: int, epsilon: float = 1.0, num_samples: int = 100
):
    """
    Uniformly sample for a d dimension ball, code from

    https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    shapes = [p.shape for p in params]
    params_flat = np.concatenate([p.detach().numpy().flatten() for p in params])

    u = np.random.normal(
        0, epsilon, (num_samples, d + 2)
    )  # an array of (d+2) normally distributed random variables
    norms = np.sum(u**2, axis=1) ** (0.5)
    for vector, norm in zip(u, norms):
        vector /= norm

    x = u[:, 0:d]  # take the first d coordinates
    x += params_flat  # (num_samples, d) + (d,) broadcasts correctly

    # Split each sample back into tensors matching original parameter shapes
    ensemble = []
    for i in range(num_samples):
        point = []
        offset = 0
        for shape in shapes:
            numel = int(np.prod(shape))
            point.append(
                torch.from_numpy(x[i, offset : offset + numel].copy())
                .float()
                .reshape(shape)
            )
            offset += numel
        ensemble.append(point)
    return ensemble
