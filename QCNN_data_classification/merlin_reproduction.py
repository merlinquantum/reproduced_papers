#!/usr/bin/env python3
# merlin_reproduction.py
# Parallel-columns only. Always use PCA-8; adapt to circuit-required input count K with a selectable adapter.

import math, random, statistics, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import perceval as pcvl
import merlin as ML
from enum import Enum

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
            lambda idx: pcvl.BS(theta=np.pi * 2 * random.random())
            // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(
                phi=np.pi * 2 * random.random()
            ),
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

def build_parallel_columns_circuit(n_modes: int, n_features: int, reservoir_mode: bool=False):
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

def required_input_params(n_modes: int, n_features: int) -> int:
    # Parallel-columns requires K = n_features * n_modes input parameters ("pl*")
    return n_features * n_modes

# ======================= Data / Utils =======================

def make_pca(k: int):
    to_t = transforms.Compose([transforms.ToTensor()])
    base_tr = datasets.MNIST("./data", train=True,  download=True, transform=to_t)
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
    random.seed(s); torch.manual_seed(s)

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

# ======================= Build Q Layer (parallel-columns only) =======================

def build_single_gi_layer(
    n_modes: int,
    n_features: int,
    shots: int = 20000,
    n_photons: int = 4,
    reservoir_mode: bool = False,
    state_pattern: str = "default"
):
    circ = build_parallel_columns_circuit(n_modes, n_features, reservoir_mode)
    trainable_prefixes = [] if reservoir_mode else ["phi"]
    input_prefixes     = ["pl"]

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
        shots=shots
    )

# ======================= Model (PCA-8 → K adapter) =======================

class SingleGI(nn.Module):
    def __init__(self, n_modes, n_features, shots: int,
                 n_photons: int, reservoir_mode: bool, state_pattern: str,
                 required_inputs: int, adapter: str):
        super().__init__()
        self.K = required_inputs
        self.adapter = adapter
        self.q = build_single_gi_layer(
            n_modes, n_features, shots=shots, n_photons=n_photons,
            reservoir_mode=reservoir_mode, state_pattern=state_pattern
        )
        # Optional learnable adapter
        if self.adapter == "linear" and self.K != 8:
            self.lin = nn.Linear(8, self.K)

    def _adapt(self, x8: torch.Tensor) -> torch.Tensor:
        if self.K == 8:
            return x8
        if self.adapter in ("auto", "slice"):
            return x8[..., :self.K] if self.K < 8 else F.pad(x8, (0, self.K - x8.shape[-1]), value=0.0)
        if self.adapter == "zero_pad":
            if self.K <= 8:
                return x8[..., :self.K]
            return F.pad(x8, (0, self.K - x8.shape[-1]), value=0.0)
        if self.adapter == "repeat":
            reps = (self.K + x8.shape[-1] - 1) // x8.shape[-1]
            y = x8.repeat(1, reps) if x8.dim() == 2 else x8
            return y[..., :self.K]
        if self.adapter == "linear":
            return self.lin(x8)
        # Fallback
        return x8[..., :self.K] if self.K < 8 else F.pad(x8, (0, self.K - x8.shape[-1]), value=0.0)

    def forward(self, angles8: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt(angles8)  # [..., K]
        return self.q(adapted)

# ======================= Train / Eval =======================

def train_once(Ztr, ytr, Zte, yte,
               n_modes, n_features, steps: int, batch: int,
               opt_name: str, lr: float, momentum: float,
               shots: int, angle_factor: float,
               n_photons: int, reservoir_mode: bool, state_pattern: str,
               required_inputs: int, adapter: str,
               seed: int) -> float:
    set_seed(seed)

    model = SingleGI(
        n_modes=n_modes, n_features=n_features, shots=shots,
        n_photons=n_photons, reservoir_mode=reservoir_mode, state_pattern=state_pattern,
        required_inputs=required_inputs, adapter=adapter
    )
    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    lossf = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=batch, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TensorDataset(Zte, yte), batch_size=512, shuffle=False)

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
    ap = argparse.ArgumentParser(description="Parallel-columns GI on MNIST 0vs1 (PCA-8 everywhere)")
    ap.add_argument("--steps", type=int, default=200, help="optimizer steps (not epochs)")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--opt", choices=["adam","sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--shots", type=int, default=20000, help="measurement shots for the GI")

    # Angle scaling
    ap.add_argument("--angle_scale", choices=["none","pi","2pi"], default="none",
                    help="map [0,1] → unchanged, [0,π], or [0,2π]")

    ap.add_argument("--n_modes", type=int, default=8)
    ap.add_argument("--n_features", type=int, default=8, help="feature count used by the circuit topology")
    ap.add_argument("--reservoir_mode", action="store_true", help="freeze interferometer params (randomized)")
    ap.add_argument("--state_pattern", choices=["default","spaced","sequential","periodic"], default="default")
    ap.add_argument("--n_photons", type=int, default=4)

    # Always PCA-8 by default
    ap.add_argument("--pca_dim", type=int, default=8)

    # Adapter from 8 → K
    ap.add_argument("--adapter", choices=["auto","slice","zero_pad","repeat","linear"], default="auto",
                    help="auto: slice if K≤8 else zero-pad; linear adds a learnable 8→K map")

    args = ap.parse_args()

    # Angle scale factor
    angle_factor = 1.0 if args.angle_scale == "none" else (math.pi if args.angle_scale == "pi" else 2 * math.pi)

    # Required K for parallel-columns
    K = required_input_params(args.n_modes, args.n_features)

    # Load PCA-8 (or user-override)
    (Ztr, ytr), (Zte, yte) = make_pca(args.pca_dim)
    print(f"PCA-{args.pca_dim} ready: train {Ztr.shape}, test {Zte.shape} | shots={args.shots}, angle={args.angle_scale}")
    print(f"Circuit: parallel_columns | n_modes={args.n_modes} n_features={args.n_features} | required inputs K={K} | adapter={args.adapter} | reservoir={args.reservoir_mode}")

    accs = []
    for s in range(args.seeds):
        print(f"[Seed {s+1}/{args.seeds}]")
        acc = train_once(
            Ztr, ytr, Zte, yte,
            args.n_modes, args.n_features,
            steps=args.steps, batch=args.batch,
            opt_name=args.opt, lr=args.lr, momentum=args.momentum,
            shots=args.shots, angle_factor=angle_factor,
            n_photons=args.n_photons, reservoir_mode=args.reservoir_mode, state_pattern=args.state_pattern,
            required_inputs=K, adapter=args.adapter,
            seed=1235 + s
        )
        print(f"  Test accuracy: {acc*100:.2f}%")
        accs.append(acc)

    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print("\n=== Summary ===")
    print("Accuracies:", ", ".join(f"{a*100:.2f}%" for a in accs))
    print(f"Mean ± Std: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    main()
