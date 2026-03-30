#!/usr/bin/env python3
# merlin_reproduction.py
# Parallel-columns only. Always use PCA-8; adapt to circuit-required input count K with a selectable adapter.

from __future__ import annotations

import argparse
import math
import random
import statistics
from enum import Enum

import merlin as ML  # noqa: N812
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ======================= State Generator =======================


class StatePattern(Enum):
    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"


class StateGenerator:
    @staticmethod
    def generate_state(n_modes, n_photons, state_pattern):
        if n_photons < 0 or n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")
        if state_pattern == StatePattern.SPACED:
            return StateGenerator._generate_spaced_state(n_modes, n_photons)
        elif state_pattern == StatePattern.SEQUENTIAL:
            return StateGenerator._generate_sequential_state(n_modes, n_photons)
        elif state_pattern in [StatePattern.PERIODIC, StatePattern.DEFAULT]:
            return StateGenerator._generate_periodic_state(n_modes, n_photons)
        else:
            return StateGenerator._generate_periodic_state(n_modes, n_photons)

    @staticmethod
    def _generate_spaced_state(n_modes, n_photons):
        if n_photons == 0:
            return [0] * n_modes
        if n_photons == 1:
            pos = n_modes // 2
            return [1 if i == pos else 0 for i in range(n_modes)]
        positions = [int(i * n_modes / n_photons) for i in range(n_photons)]
        positions = [min(pos, n_modes - 1) for pos in positions]
        occ = [0] * n_modes
        for pos in positions:
            occ[pos] += 1
        return occ

    @staticmethod
    def _generate_periodic_state(n_modes, n_photons):
        bits = [1 if i % 2 == 0 else 0 for i in range(min(n_photons * 2, n_modes))]
        count = sum(bits)
        i = 0
        while count < n_photons and i < n_modes:
            if i >= len(bits):
                bits.append(0)
            if bits[i] == 0:
                bits[i] = 1
                count += 1
            i += 1
        padding = [0] * (n_modes - len(bits))
        return bits + padding

    @staticmethod
    def _generate_sequential_state(n_modes, n_photons):
        return [1 if i < n_photons else 0 for i in range(n_modes)]


# ======================= Parallel-Columns Circuit Only =======================


def _generate_interferometer(n_modes, stage_idx, reservoir_mode=False):
    if reservoir_mode:
        return pcvl.GenericInterferometer(
            n_modes,
            lambda idx: (
                pcvl.BS(theta=np.pi * 2 * random.random())
                // (0, pcvl.PS(phi=np.pi * 2 * random.random()))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )
    else:

        def mzi(P1, P2):
            return (
                pcvl.Circuit(2)
                .add((0, 1), pcvl.BS())
                .add(0, pcvl.PS(P1))
                .add((0, 1), pcvl.BS())
                .add(0, pcvl.PS(P2))
            )

    offset = stage_idx * (n_modes * (n_modes - 1) // 2)
    shape = pcvl.InterferometerShape.RECTANGLE
    return pcvl.GenericInterferometer(
        n_modes,
        fun_gen=lambda idx: mzi(
            pcvl.P(f"phi_0{offset + idx}"), pcvl.P(f"phi_1{offset + idx}")
        ),
        shape=shape,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(
            phi=pcvl.P(f"phi_02{stage_idx}_{idx}")
        ),
    )


def build_parallel_columns_circuit(
    n_modes: int, n_features: int, reservoir_mode: bool = False
):
    """
    Columns of: [Interferometer] -> [per-mode PS (trainable inputs)] repeated n_features times,
    then a final [Interferometer].
    """
    circuit = pcvl.Circuit(n_modes)
    ps_idx = 0
    for stage in range(n_features + 1):
        circuit.add(0, _generate_interferometer(n_modes, stage, reservoir_mode))
        if stage < n_features:
            for m_idx in range(n_modes):
                circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
                ps_idx += 1
    return circuit


def create_quantum_circuit(n_modes: int, n_features: int):
    # 1. Left interferometer - trainable transformation
    wl = pcvl.GenericInterferometer(
        n_modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_li{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_lo{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    # 2. Input encoding - maps classical data to quantum parameters
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_features):  # 4 input features
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i, pcvl.PS(px))
    # 3. Right interferometer - trainable transformation
    wr = pcvl.GenericInterferometer(
        n_modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ri{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ro{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    # Combine all components
    return wl // c_var // wr


def required_input_params(n_modes: int, n_features: int) -> int:
    if n_features > n_modes:
        raise ValueError(
            "n_features cannot exceed n_modes when matching inputs one-to-one."
        )
    return n_features


# ======================= Data / Utils =======================


def make_pca(k: int):
    to_t = transforms.Compose([transforms.ToTensor()])
    base_tr = datasets.MNIST("./data", train=True, download=True, transform=to_t)
    base_te = datasets.MNIST("./data", train=False, download=True, transform=to_t)

    def filt(base):
        Xs, Ys = [], []
        for img, lab in base:
            if int(lab) in (0, 1):
                Xs.append(img.view(-1).float())
                Ys.append(0 if int(lab) == 0 else 1)
        return torch.stack(Xs, 0), torch.tensor(Ys, dtype=torch.long)

    Xtr, ytr = filt(base_tr)
    Xte, yte = filt(base_te)
    pca = PCA(n_components=k, svd_solver="full", whiten=False, random_state=0)
    Ztr_raw = torch.from_numpy(pca.fit_transform(Xtr.numpy())).float()
    Zte_raw = torch.from_numpy(pca.transform(Xte.numpy())).float()
    mins = Ztr_raw.min(0, keepdim=True).values
    maxs = Ztr_raw.max(0, keepdim=True).values
    Ztr = torch.clamp((Ztr_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    Zte = torch.clamp((Zte_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    return (Ztr, ytr), (Zte, yte)


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, angle_factor: float) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        logits = model(xb * angle_factor)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


###############################
### Naive approach using GI ###
###############################

# ======================= Build Q Layer (parallel-columns only) =======================


def build_single_gi_layer(
    n_modes: int,
    n_features: int,
    n_photons: int = 4,
    reservoir_mode: bool = False,
    state_pattern: str = "default",
):
    # circ = build_parallel_columns_circuit(n_modes, n_features, reservoir_mode)
    circ = create_quantum_circuit(n_modes, n_features)
    # pcvl.pdisplay(circ)
    trainable_prefixes = [] if reservoir_mode else ["theta"]
    input_prefixes = ["px"]

    print(
        f"N_modes = {n_modes} and n_photons = {n_photons} and n_features = {n_features}"
    )
    pat = StatePattern[state_pattern.upper()] if state_pattern else StatePattern.DEFAULT
    input_state = StateGenerator.generate_state(n_modes, n_photons, pat)

    return ML.QuantumLayer(
        input_size=n_modes,
        output_size=2,
        circuit=circ,
        trainable_parameters=trainable_prefixes,
        input_parameters=input_prefixes,
        input_state=input_state,
        output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
    )


# ======================= Model (PCA-8 → K adapter) =======================


class SingleGI(nn.Module):
    def __init__(
        self,
        n_modes,
        n_features,
        n_photons: int,
        reservoir_mode: bool,
        state_pattern: str,
        required_inputs: int,
        adapter: str,
        input_dim: int,
    ):
        super().__init__()
        self.K = required_inputs
        self.input_dim = input_dim
        if self.input_dim != self.K:
            raise ValueError(
                f"Quantum layer expects {self.K} features but received {self.input_dim}. "
                "Set circuit and preprocessing parameters so they match exactly."
            )
        self.q = build_single_gi_layer(
            n_modes,
            n_features,
            n_photons=n_photons,
            reservoir_mode=reservoir_mode,
            state_pattern=state_pattern,
        )

    def _adapt(self, features: torch.Tensor) -> torch.Tensor:
        """
        Identity pass-through kept for backward compatibility with previous API.
        Validates shapes to ensure the quantum circuit receives exactly K inputs.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        current_dim = features.shape[-1]
        if current_dim != self.K:
            raise ValueError(
                f"Expected feature width {self.K}, received {current_dim}. "
                "Adjust preprocessing so the feature dimension matches the circuit requirements."
            )
        return features

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt(angles)  # [..., K]
        return self.q(adapted)


###################################
### Pseudo-convolution approach ###
###################################


class QuantumPatchKernel(nn.Module):
    """Wrap a QuantumLayer so it can process sliding PCA patches."""

    def __init__(
        self,
        q_layer: ML.QuantumLayer,
        required_inputs: int,
        patch_dim: int,
    ):
        super().__init__()
        self.q = q_layer
        self.K = required_inputs
        self.patch_dim = patch_dim
        if self.patch_dim != self.K:
            raise ValueError(
                f"Quantum kernel expects patch size {self.K}, received {self.patch_dim}. "
                "Ensure kernel_size equals the number of encoded quantum features."
            )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.dim() == 1:
            patch = patch.unsqueeze(0)
        if patch.shape[-1] != self.K:
            raise ValueError(
                f"Expected patch with last dimension {self.K}, got {patch.shape[-1]}."
            )
        return self.q(patch)

    @property
    def output_size(self) -> int:
        if getattr(self.q, "output_size", None) is None:
            raise ValueError("QuantumLayer does not expose output_size attribute.")
        return self.q.output_size


class QConvModel(nn.Module):
    """
    Apply a set of quantum (or classical) kernels in a convolutional manner over PCA components,
    then aggregate the flattened responses with a linear head.
    """

    def __init__(
        self,
        input_dim: int,
        n_kernels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        kernel_modules: list[nn.Module] | None = None,
    ):
        super().__init__()
        if n_kernels <= 0:
            raise ValueError("n_kernels must be a positive integer.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_quantum = kernel_modules is not None
        if self.use_quantum:
            if kernel_modules is None or len(kernel_modules) != n_kernels:
                raise ValueError("kernel_modules must contain n_kernels modules.")
            self.kernel_modules = nn.ModuleList(kernel_modules)
            self.kernel_output_dim = getattr(
                self.kernel_modules[0], "output_size", None
            )
            if self.kernel_output_dim is None:
                raise ValueError(
                    "Quantum kernels must expose an output_size attribute."
                )
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            weight = torch.empty(n_kernels, kernel_size)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weight = nn.Parameter(weight)
            if bias:
                bound = 1 / math.sqrt(kernel_size)
                self.bias = nn.Parameter(torch.empty(n_kernels).uniform_(-bound, bound))
            else:
                self.register_parameter("bias", None)
            self.kernel_modules = None
            self.kernel_output_dim = 1

        self.conv_output_dim = self.output_dim(input_dim)
        self.head = nn.Linear(self.conv_output_dim, 2)

    def _compute_num_windows(self, input_len: int) -> int:
        if input_len < self.kernel_size:
            raise ValueError(
                f"Input length {input_len} is smaller than kernel size {self.kernel_size}."
            )
        return 1 + (input_len - self.kernel_size) // self.stride

    def output_dim(self, input_len: int) -> int:
        """Number of features produced for a given PCA dimension."""
        num_windows = self._compute_num_windows(input_len)
        return num_windows * self.n_kernels * self.kernel_output_dim

    def _apply_classical_kernels(self, patches: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bpk,nk->bpn", patches, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        out = out.permute(0, 2, 1).contiguous()
        return out.view(out.size(0), -1)

    def _apply_quantum_kernels(
        self, patches: torch.Tensor, num_windows: int, batch_size: int
    ) -> torch.Tensor:
        patches_flat = patches.contiguous().view(-1, patches.size(-1))
        outputs = []
        for kernel in self.kernel_modules:
            y = kernel(patches_flat)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.view(batch_size, num_windows, -1)
            outputs.append(y)
        out = torch.stack(outputs, dim=1)  # (batch, n_kernels, num_windows, out_dim)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                "QConvModel expects inputs of shape (batch, n_components) "
                "or (n_components,) optionally with a singleton channel dimension."
            )

        patches = x.unfold(dimension=-1, size=self.kernel_size, step=self.stride)
        num_windows = patches.size(1)
        if num_windows == 0:
            raise ValueError(
                "Unfold produced zero windows. Check kernel_size and stride values."
            )

        if self.use_quantum:
            features = self._apply_quantum_kernels(patches, num_windows, x.size(0))
        else:
            features = self._apply_classical_kernels(patches)

        return self.head(features)


# ======================= Train / Eval =======================


def train_once(
    Ztr,
    ytr,
    Zte,
    yte,
    n_modes,
    n_features,
    steps: int,
    batch: int,
    opt_name: str,
    lr: float,
    momentum: float,
    angle_factor: float,
    n_photons: int,
    reservoir_mode: bool,
    state_pattern: str,
    required_inputs: int,
    adapter: str,
    seed: int,
    model: nn.Module,
) -> float:
    set_seed(seed)

    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optim = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=True
        )
    lossf = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(Ztr, ytr), batch_size=batch, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(TensorDataset(Zte, yte), batch_size=512, shuffle=False)

    model.train()
    it = 0
    while it < steps:
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            logits = model(xb * angle_factor)
            loss = lossf(logits, yb)
            loss.backward()
            optim.step()
            it += 1
            if it >= steps:
                break

    return evaluate(model, test_loader, angle_factor)


# ======================= CLI =======================


def main():
    ap = argparse.ArgumentParser(
        description="Parallel-columns GI on MNIST 0vs1 (PCA-8 everywhere)"
    )
    ap.add_argument(
        "--steps", type=int, default=200, help="optimizer steps (not epochs)"
    )
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)

    # Angle scaling
    ap.add_argument(
        "--angle_scale",
        choices=["none", "pi", "2pi"],
        default="none",
        help="map [0,1] → unchanged, [0,π], or [0,2π]",
    )

    ap.add_argument("--n_modes", type=int, default=8)
    ap.add_argument(
        "--n_features",
        type=int,
        default=8,
        help="feature count used by the circuit topology",
    )
    ap.add_argument(
        "--reservoir_mode",
        action="store_true",
        help="freeze interferometer params (randomized)",
    )
    ap.add_argument(
        "--state_pattern",
        choices=["default", "spaced", "sequential", "periodic"],
        default="default",
    )
    ap.add_argument("--n_photons", type=int, default=4)

    # Always PCA-8 by default
    ap.add_argument("--pca_dim", type=int, default=8)

    ap.add_argument(
        "--model",
        choices=["qconv", "single"],
        default="qconv",
        help="choose between quantum convolution (default) or single GI model",
    )

    # Adapter from 8 → K
    ap.add_argument(
        "--adapter",
        choices=["auto", "slice", "zero_pad", "repeat", "linear"],
        default="auto",
        help="auto: slice if K≤8 else zero-pad; linear adds a learnable 8→K map",
    )
    ap.add_argument(
        "--nb_kernels",
        type=int,
        default=4,
        help="number of kernels for the pseudo-convolution",
    )
    ap.add_argument(
        "--kernel_size",
        type=int,
        default=2,
        help="kernel size for the pseudo-convolution",
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="stride for the pseudo-convolution"
    )
    ap.add_argument(
        "--conv_classical",
        action="store_true",
        help="use classical learnable kernels instead of QuantumLayer kernels",
    )
    ap.add_argument(
        "--compare_classical",
        action="store_true",
        help="run both quantum and classical qconv variants with matching kernel counts",
    )
    ap.add_argument(
        "--kernel_modes",
        type=int,
        default=8,
        help="number of optical modes per quantum kernel (default: kernel_size)",
    )

    args = ap.parse_args()

    # Angle scale factor
    angle_factor = (
        1.0
        if args.angle_scale == "none"
        else (math.pi if args.angle_scale == "pi" else 2 * math.pi)
    )

    # Required K for parallel-columns
    K = required_input_params(args.n_modes, args.n_features)

    _conv_output_dim = None
    _conv_mode_str = "disabled"
    if args.compare_classical and args.conv_classical:
        raise ValueError("compare_classical cannot be combined with conv_classical.")

    if args.model == "qconv":
        if args.kernel_size > args.pca_dim:
            raise ValueError("kernel_size cannot exceed the PCA dimension.")
        if args.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        num_windows = 1 + (args.pca_dim - args.kernel_size) // args.stride
        if num_windows <= 0:
            raise ValueError("qconv configuration results in zero sliding windows.")
        base_conv_params = {
            "n_kernels": args.nb_kernels,
            "kernel_size": args.kernel_size,
            "stride": args.stride,
        }
        classical_conv_params = {**base_conv_params, "bias": True}
        classical_output_dim = num_windows * args.nb_kernels
        quantum_conv_params = None
        quantum_output_dim = None
        conv_mode_logs = []

        use_quantum = not args.conv_classical or args.compare_classical
        if use_quantum:
            kernel_modes = args.kernel_modes or args.kernel_size
            if kernel_modes <= 0:
                raise ValueError("kernel_modes must be a positive integer.")
            kernel_required_inputs = required_input_params(
                kernel_modes, args.kernel_size
            )
            print(
                f"Building quantum kernels on {kernel_modes} modes with {args.kernel_size} features | "
                f"{args.nb_kernels} kernels, stride={args.stride}"
            )

            def _make_kernel() -> QuantumPatchKernel:
                q_layer = ML.QuantumLayer(
                    input_size=kernel_modes,
                    output_size=2,
                    circuit=create_quantum_circuit(kernel_modes, args.kernel_size),
                    trainable_parameters=["theta"],
                    input_parameters=["px"],
                    input_state=([1, 0] * (kernel_modes // 2) + [0])
                    if kernel_modes % 2 == 1
                    else [1, 0] * (kernel_modes // 2),
                    output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
                )
                return QuantumPatchKernel(
                    q_layer,
                    required_inputs=kernel_required_inputs,
                    patch_dim=args.kernel_size,
                )

            sample_kernel = _make_kernel()
            sample_out_dim = sample_kernel.output_size
            del sample_kernel

            def _factory():
                return [_make_kernel() for _ in range(args.nb_kernels)]

            quantum_conv_params = {
                **base_conv_params,
                "bias": False,
                "kernel_modules_factory": _factory,
            }
            quantum_output_dim = num_windows * args.nb_kernels * sample_out_dim
            conv_mode_logs.append(
                f"quantum: {args.nb_kernels} kernels → {quantum_output_dim} dims (out={sample_out_dim})"
            )

        if args.conv_classical or args.compare_classical or not use_quantum:
            conv_mode_logs.append(
                f"classical: {args.nb_kernels} kernels → {classical_output_dim} dims"
            )

    # Load PCA-8 (or user-override)
    (Ztr, ytr), (Zte, yte) = make_pca(args.pca_dim)
    print(
        f"PCA-{args.pca_dim} ready: train {Ztr.shape}, test {Zte.shape} | angle={args.angle_scale}"
    )
    if args.model == "qconv" and conv_mode_logs:
        print(
            f"Model: qconv | circuit=parallel_columns | n_modes={args.n_modes} "
            f"n_features={args.n_features} | required inputs K={K} | adapter={args.adapter} "
            f"| reservoir={args.reservoir_mode}"
        )
        print("Convolution configurations (matching kernel count):")
        for log_entry in conv_mode_logs:
            print(f"  - {log_entry}")
    else:
        print(
            f"Model: {args.model} | circuit=parallel_columns | n_modes={args.n_modes} "
            f"n_features={args.n_features} | required inputs K={K} | adapter={args.adapter} "
            f"| reservoir={args.reservoir_mode}"
        )

    input_dim = Ztr.shape[-1]
    model_variants = []
    if args.model == "single":

        def build_single() -> nn.Module:
            return SingleGI(
                n_modes=args.n_modes,
                n_features=args.n_features,
                n_photons=args.n_photons,
                reservoir_mode=args.reservoir_mode,
                state_pattern=args.state_pattern,
                required_inputs=K,
                adapter=args.adapter,
                input_dim=input_dim,
            )

        model_variants.append(("single", build_single))
    elif args.model == "qconv":
        if use_quantum and quantum_conv_params is not None:

            def build_quantum() -> nn.Module:
                params = dict(quantum_conv_params)
                factory = params.pop("kernel_modules_factory", None)
                if factory is not None:
                    params["kernel_modules"] = factory()
                return QConvModel(input_dim=input_dim, **params)

            model_variants.append(("qconv_quantum", build_quantum))

        if args.conv_classical or args.compare_classical or not use_quantum:

            def build_classical() -> nn.Module:
                return QConvModel(input_dim=input_dim, **classical_conv_params)

            label = (
                "qconv_classical"
                if (args.compare_classical or use_quantum)
                else "qconv_classical_only"
            )
            model_variants.append((label, build_classical))
    else:
        raise ValueError(f"Unhandled model type: {args.model}")

    comparison_results = []
    for variant_name, builder in model_variants:
        print(
            f"\n=== Evaluating {variant_name} ({args.seeds} seed{'s' if args.seeds > 1 else ''}) ==="
        )
        variant_accs = []
        for s in range(args.seeds):
            print(f"[Seed {s + 1}/{args.seeds}]")
            model = builder()
            print(
                f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
            )
            acc = train_once(
                Ztr,
                ytr,
                Zte,
                yte,
                args.n_modes,
                args.n_features,
                steps=args.steps,
                batch=args.batch,
                opt_name=args.opt,
                lr=args.lr,
                momentum=args.momentum,
                angle_factor=angle_factor,
                n_photons=args.n_photons,
                reservoir_mode=args.reservoir_mode,
                state_pattern=args.state_pattern,
                required_inputs=K,
                adapter=args.adapter,
                seed=1235 + s,
                model=model,
            )
            print(f"  Test accuracy: {acc * 100:.2f}%")
            variant_accs.append(acc)
        mean = statistics.mean(variant_accs)
        std = statistics.stdev(variant_accs) if len(variant_accs) > 1 else 0.0
        print(
            f"→ Summary for {variant_name}: mean {mean * 100:.2f}% ± {std * 100:.2f}%"
        )
        comparison_results.append((variant_name, variant_accs, mean, std))

    if len(comparison_results) > 1:
        print("\n=== Overall Comparison ===")
        for name, accs, mean, std in comparison_results:
            acc_line = ", ".join(f"{a * 100:.2f}%" for a in accs)
            print(f"{name}: [{acc_line}] → mean {mean * 100:.2f}% ± {std * 100:.2f}%")
    elif comparison_results:
        name, accs, mean, std = comparison_results[0]
        print("\n=== Summary ===")
        print("Accuracies:", ", ".join(f"{a * 100:.2f}%" for a in accs))
        print(f"Mean ± Std: {mean * 100:.2f}% ± {std * 100:.2f}%")


if __name__ == "__main__":
    main()
