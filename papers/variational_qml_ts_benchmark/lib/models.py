"""Quantum and classical time-series models.

Faithful port of ``utils/models.py`` from
tobias-fllnr/VariationalQMLTimeSeriesBenchmark, covering the five quantum models
(d-QNN, ru-QNN, QRNN, QLSTM, le-QLSTM) and three classical baselines
(MLP, RNN, LSTM) benchmarked in arXiv:2504.12416.

Bug fixes relative to the original code are gated behind the ``bugfix`` flag and
documented in ``BUGS.md``.  With ``bugfix=False`` the models reproduce the
original behaviour exactly.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _output_dim(data_label: str) -> int:
    if data_label.startswith("lorenz"):
        return 3
    if data_label.startswith("henon"):
        return 2
    return 1


class VQC(nn.Module):
    """QNN models: dressed QNN (d-QNN) and re-uploading QNN (ru-QNN).

    ``ansatz`` selects the variant:
      * ``"paper_rivera-ruiz_with_inputlayer_<L>"`` -> d-QNN with L variational
        layers, an input linear layer (l*d -> n qubits) and RY angle encoding.
      * ``"ruexp_<blocks>"`` -> ru-QNN with exponential re-uploading encoding.
    """

    def __init__(
        self,
        num_qubits: int,
        seq_length: int,
        ansatz: str,
        data_label: str,
        random_id: int = 42,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        bugfix: bool = False,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method
        self.bugfix = bugfix

        torch.manual_seed(self.random_id)
        self.weight_init = {
            "weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)
        }
        self.dev = qml.device(self.backend, wires=num_qubits)
        self.vqc_torch_layer = self._build_vqc()

        out_dim = _output_dim(data_label)
        self.output_layer = nn.Linear(num_qubits, out_dim)

        self.ansatz_input_layer_start = "paper_rivera-ruiz_with_inputlayer_"
        if self.ansatz.startswith(self.ansatz_input_layer_start):
            in_dim = _output_dim(data_label) * self.seq_length
            self.input_layer = nn.Linear(in_dim, self.num_qubits)

        self.vqc_torch_layer.weights.requires_grad = True

    def _build_vqc(self) -> nn.Module:
        if self.ansatz.startswith("paper_rivera-ruiz_with_inputlayer_"):
            num_layers = int(self.ansatz.split("_")[-1])

            @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
            def circuit(inputs, weights):
                for i in range(self.num_qubits):
                    qml.RY(torch.pi * inputs[:, i], wires=i)
                for j in range(num_layers):
                    for i in range(self.num_qubits):
                        qml.RX(weights[i][j][0], wires=i)
                        qml.RY(weights[i][j][1], wires=i)
                        qml.RZ(weights[i][j][2], wires=i)
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

            weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}
            return qml.qnn.TorchLayer(
                circuit, weight_shapes, init_method=self.weight_init
            )

        elif self.ansatz.startswith("ruexp_"):
            strings = self.ansatz.split("_")[1:]
            parameter_blocks_count = sum(1 for s in strings if not s.startswith("E"))
            n = self.num_qubits

            def block(inputs, weights, i):
                enc_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
                pc = 0
                for s in strings:
                    if s.startswith("E"):
                        c1, c2 = s[1], s[2]
                        input_col = None
                        if self.data_label.startswith("lorenz"):
                            off = {"X": 0, "Y": 1, "Z": 2}.get(c1)
                            if off is not None:
                                input_col = 3 * i + off
                        elif self.data_label.startswith("henon"):
                            off = {"X": 0, "Y": 1}.get(c1)
                            if off is not None:
                                input_col = 2 * i + off
                        else:
                            if c1 == "X":
                                input_col = i
                        rot = enc_gate.get(c2)
                        if input_col is not None and rot is not None:
                            for j in range(n):
                                # Exponential encoding prefactor.
                                # BUG (original code): 3**(j-n), so the top qubit
                                # gets 1/3 instead of the paper's beta_a=3^(a-1)/3^(n-1)
                                # which reaches 1 for the top qubit.  See BUGS.md #2.
                                exponent = (j - n + 1) if self.bugfix else (j - n)
                                angle = torch.pi * inputs[:, input_col] * (3**exponent)
                                rot(angle, wires=j)
                    elif s in ("X", "Y", "Z"):
                        g = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[s]
                        for j in range(n):
                            g(weights[i][j][pc], wires=j)
                        pc += 1
                    elif s in ("CX", "CY", "CZ"):
                        cg = {"CX": qml.CRX, "CY": qml.CRY, "CZ": qml.CRZ}[s]
                        for j in range(n - 1):
                            cg(weights[i][j][pc], wires=[j, j + 1])
                        cg(weights[i][n - 1][pc], wires=[n - 1, 0])
                        pc += 1

            @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
            def circuit(inputs, weights):
                for i in range(self.seq_length):
                    block(inputs, weights, i)
                return [qml.expval(qml.PauliZ(i)) for i in range(n)]

            weight_shapes = {"weights": (self.seq_length, n, parameter_blocks_count)}
            return qml.qnn.TorchLayer(
                circuit, weight_shapes, init_method=self.weight_init
            )

        raise ValueError(f"Unknown VQC ansatz: {self.ansatz!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (x.size(0), -1))
        if self.ansatz.startswith(self.ansatz_input_layer_start):
            x = self.input_layer(x)
        return self.output_layer(self.vqc_torch_layer(x))


def _qlstm_pqc_factory(dev, num_qubits, num_layers, diff_method, bugfix=False):
    """Return the QLSTM PQC circuit function (shared by QLSTM and le-QLSTM).

    Original code (bugfix=False) applies the nearest-neighbour CNOT ring twice
    and uses ``qml.Rot`` (= Rz Ry Rz).  The paper (App. A, Fig. 9b) specifies
    *nearest and second-nearest* neighbour CNOTs and a Rz Rx Rz rotation.  The
    fixed path (bugfix=True) implements the paper faithfully.  See BUGS.md #3/#5.
    """

    def VQC(inputs, weights):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.RY(torch.arctan(inputs[:, i]), wires=i)
            qml.RZ(torch.arctan(inputs[:, i] ** 2), wires=i)
        for j in range(num_layers):
            # First entangling ring: nearest neighbours.
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[num_qubits - 1, 0])
            # Second ring.
            if bugfix:
                # Second-nearest neighbours (paper).
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 2) % num_qubits])
            else:
                # Original code: a duplicate nearest-neighbour ring.
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[num_qubits - 1, 0])
            for i in range(num_qubits):
                if bugfix:
                    qml.RZ(weights[i][j][0], wires=i)
                    qml.RX(weights[i][j][1], wires=i)
                    qml.RZ(weights[i][j][2], wires=i)
                else:
                    qml.Rot(
                        weights[i][j][0], weights[i][j][1], weights[i][j][2], wires=i
                    )
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

    return qml.QNode(VQC, dev, diff_method=diff_method, interface="torch")


class QLSTM_Paper(nn.Module):
    """Original QLSTM (Chen et al. 2022) with six PQCs per cell."""

    def __init__(
        self,
        num_qubits: int,
        ansatz: str,
        data_label: str,
        random_id: int = 42,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        bugfix: bool = False,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method
        self.bugfix = bugfix

        self.output_layer = nn.Linear(num_qubits, _output_dim(data_label))
        torch.manual_seed(self.random_id)
        self.weight_init = {
            "weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)
        }
        self.dev = qml.device(self.backend, wires=num_qubits)
        num_layers = int(self.ansatz.split("_")[-1])
        weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}

        self.vqcs = nn.ModuleList()
        for _ in range(6):
            node = _qlstm_pqc_factory(
                self.dev, num_qubits, num_layers, self.diff_method, bugfix=self.bugfix
            )
            self.vqcs.append(
                qml.qnn.TorchLayer(node, weight_shapes, init_method=self.weight_init)
            )
        self.vqc1, self.vqc2, self.vqc3, self.vqc4, self.vqc5, self.vqc6 = self.vqcs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, features_size = x.size()
        h_t = torch.zeros(batch_size, self.num_qubits - features_size)
        c_t = torch.zeros(batch_size, self.num_qubits)
        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((x_t, h_t), dim=1)
            f_t = torch.sigmoid(self.vqc1(v_t))
            i_t = torch.sigmoid(self.vqc2(v_t))
            g_t = torch.tanh(self.vqc3(v_t))
            o_t = torch.sigmoid(self.vqc4(v_t))
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = self.vqc5(o_t * torch.tanh(c_t))[:, : self.num_qubits - features_size]
        y_t = self.vqc6(o_t * torch.tanh(c_t))
        return self.output_layer(y_t)


class QLSTM_Linear_Enhanced_Paper(nn.Module):
    """Linear-layer-enhanced QLSTM (Cao et al. 2023): four PQCs + linear layers."""

    def __init__(
        self,
        num_qubits: int,
        hidden_size: int,
        ansatz: str,
        data_label: str,
        random_id: int = 42,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        bugfix: bool = False,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method
        self.bugfix = bugfix

        out_dim = _output_dim(data_label)
        self.output_layer = nn.Linear(self.hidden_size, out_dim)
        self.input_size = out_dim

        torch.manual_seed(self.random_id)
        self.weight_init = {
            "weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)
        }
        self.dev = qml.device(self.backend, wires=num_qubits)
        num_layers = int(self.ansatz.split("_")[-1])
        weight_shapes = {"weights": (self.num_qubits, num_layers, 3)}

        self.vqcs = nn.ModuleList()
        for _ in range(4):
            node = _qlstm_pqc_factory(
                self.dev, num_qubits, num_layers, self.diff_method, bugfix=self.bugfix
            )
            self.vqcs.append(
                qml.qnn.TorchLayer(node, weight_shapes, init_method=self.weight_init)
            )
        self.vqc1, self.vqc2, self.vqc3, self.vqc4 = self.vqcs

        self.linear_in = nn.Linear(self.hidden_size + self.input_size, self.num_qubits)
        self.linear_out_1 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_2 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_3 = nn.Linear(self.num_qubits, self.hidden_size)
        self.linear_out_4 = nn.Linear(self.num_qubits, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = self.linear_in(torch.cat((x_t, h_t), dim=1))
            f_t = torch.sigmoid(self.linear_out_1(self.vqc1(v_t)))
            i_t = torch.sigmoid(self.linear_out_2(self.vqc2(v_t)))
            g_t = torch.tanh(self.linear_out_3(self.vqc3(v_t)))
            o_t = torch.sigmoid(self.linear_out_4(self.vqc4(v_t)))
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
        return self.output_layer(h_t)


class QRNN_Paper(nn.Module):
    """Quantum RNN (Li et al. 2023), data + hidden register, shared weights.

    The benchmark uses ``ansatz="paper_no_reset"`` (data register is *not* reset
    between timesteps); ``"paper_reset"`` reproduces the original mid-circuit
    reset variant used only for the small-system ablation (Appendix B).
    """

    def __init__(
        self,
        num_qubits: int,
        num_qubits_hidden: int,
        seq_length: int,
        ansatz: str,
        data_label: str,
        random_id: int = 42,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        bugfix: bool = False,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        self.random_id = random_id
        self.backend = backend
        self.diff_method = diff_method
        self.bugfix = bugfix
        self.num_qubits_hidden = num_qubits_hidden
        self.num_qubits_data = num_qubits - num_qubits_hidden

        self.output_layer = nn.Linear(self.num_qubits_data, _output_dim(data_label))
        torch.manual_seed(self.random_id)
        self.weight_init = {
            "weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)
        }
        self.dev = qml.device(self.backend, wires=num_qubits)
        self.vqc_torch_layer = self._build_vqc()

    def _encode(self, inputs, i):
        n_data = self.num_qubits_data
        if self.data_label.startswith("lorenz"):
            for j in range(n_data):
                qml.RY(torch.arccos(inputs[:, 3 * i]), wires=j)
                qml.RX(torch.arccos(inputs[:, 1 + 3 * i]), wires=j)
                qml.RY(torch.arccos(inputs[:, 2 + 3 * i]), wires=j)
        elif self.data_label.startswith("henon"):
            for j in range(n_data):
                qml.RX(torch.arccos(inputs[:, 2 * i]), wires=j)
                qml.RY(torch.arccos(inputs[:, 1 + 2 * i]), wires=j)
        else:
            for j in range(n_data):
                qml.RY(torch.arccos(inputs[:, i]), wires=j)

    def _variational(self, weights):
        n = self.num_qubits
        for j in range(n):
            qml.RX(weights[j][0], wires=j)
            qml.RZ(weights[j][1], wires=j)
            qml.RX(weights[j][2], wires=j)
        for j in range(n - 1):
            qml.CNOT(wires=[j, j + 1])
            qml.RZ(weights[j + 1][3], wires=j + 1)
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[n - 1, 0])
        qml.RZ(weights[0][3], wires=0)
        qml.CNOT(wires=[n - 1, 0])

    def _build_vqc(self) -> nn.Module:
        reset = self.ansatz == "paper_reset"
        n_data = self.num_qubits_data

        @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
        def circuit(inputs, weights):
            for i in range(self.seq_length):
                self._encode(inputs, i)
                self._variational(weights)
                if reset:
                    if i == self.seq_length - 1:
                        return [qml.expval(qml.PauliZ(k)) for k in range(n_data)]
                    for j in range(n_data):
                        qml.measure(wires=j, reset=True)
            return [qml.expval(qml.PauliZ(k)) for k in range(n_data)]

        weight_shapes = {"weights": (self.num_qubits, 4)}
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_label.startswith(("lorenz", "henon")):
            x = torch.reshape(x, (x.size(0), -1))
        else:
            x = x.squeeze(-1)
        return self.output_layer(self.vqc_torch_layer(x))


class LSTM(nn.Module):
    def __init__(self, hidden_size, ansatz, data_label, random_id=42, **_):
        super().__init__()
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.num_layers = int(ansatz.split("_")[-1])
        torch.manual_seed(random_id)
        d = _output_dim(data_label)
        self.lstm = nn.LSTM(d, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, d)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RNN(nn.Module):
    def __init__(self, hidden_size, ansatz, data_label, random_id=42, **_):
        super().__init__()
        self.hidden_size = hidden_size
        self.ansatz = ansatz
        self.data_label = data_label
        self.num_layers = int(ansatz.split("_")[-1])
        torch.manual_seed(random_id)
        d = _output_dim(data_label)
        self.rnn = nn.RNN(d, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, d)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class MLP(nn.Module):
    def __init__(self, seq_length, ansatz, data_label, random_id=42, **_):
        super().__init__()
        self.seq_length = seq_length
        self.ansatz = ansatz
        self.data_label = data_label
        torch.manual_seed(random_id)
        self.activation, self.hidden_sizes = self._parse_ansatz()
        d = _output_dim(data_label)
        self.input_size = d * seq_length
        self.output_size = d
        act = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}[
            self.activation
        ]()

        layers = [nn.Linear(self.input_size, self.hidden_sizes[0]), act]
        for i in range(1, len(self.hidden_sizes)):
            layers += [nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]), act]
        layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        self.mlp_layers = nn.Sequential(*layers)

    def _parse_ansatz(self) -> tuple[str, list[int]]:
        parts = self.ansatz.split("_")
        return parts[0], [int(p) for p in parts[1:]]

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        return self.mlp_layers(x)


def build_model(
    model_name: str,
    data_label: str,
    seq_length: int,
    ansatz: str,
    num_qubits: int | None,
    hidden_size: int | None,
    random_id: int,
    bugfix: bool = False,
) -> nn.Module:
    """Factory mirroring ``training_and_analyzing.train_and_analyse``."""
    if model_name in ("vqc", "d_qnn", "ru_qnn"):
        return VQC(num_qubits, seq_length, ansatz, data_label, random_id, bugfix=bugfix)
    if model_name in ("qlstm_paper", "qlstm"):
        return QLSTM_Paper(num_qubits, ansatz, data_label, random_id, bugfix=bugfix)
    if model_name in ("qlstm_linear_enhanced_paper", "le_qlstm"):
        return QLSTM_Linear_Enhanced_Paper(
            num_qubits, hidden_size, ansatz, data_label, random_id, bugfix=bugfix
        )
    if model_name in ("qrnn_paper", "qrnn"):
        return QRNN_Paper(
            num_qubits,
            hidden_size,
            seq_length,
            ansatz,
            data_label,
            random_id,
            bugfix=bugfix,
        )
    if model_name == "lstm":
        return LSTM(hidden_size, ansatz, data_label, random_id)
    if model_name == "rnn":
        return RNN(hidden_size, ansatz, data_label, random_id)
    if model_name == "mlp":
        return MLP(seq_length, ansatz, data_label, random_id)
    if model_name in ("photonic", "photonic_dqnn"):
        from .photonic import PhotonicDressedQNN

        return PhotonicDressedQNN(
            data_label,
            seq_length,
            n_modes=(num_qubits if num_qubits else 6),
            n_photons=(hidden_size if hidden_size else 3),
            random_id=random_id,
        )
    # Non-variational photonic reservoirs (extension, outside the paper's scope
    # -- see lib/reservoir.py). Only the linear readout trains.
    if model_name in ("photonic_reservoir", "reservoir"):
        from .reservoir import PhotonicReservoir, parse_reservoir_ansatz

        hp = parse_reservoir_ansatz(ansatz)
        return PhotonicReservoir(
            data_label,
            seq_length,
            n_modes=(num_qubits if num_qubits else 6),
            n_photons=(hidden_size if hidden_size else 3),
            random_id=random_id,
            **{k: v for k, v in hp.items() if k == "scale"},
        )
    if model_name in ("photonic_memristor", "memristor"):
        from .reservoir import PhotonicMemristiveReservoir, parse_reservoir_ansatz

        hp = parse_reservoir_ansatz(ansatz)
        kw = {k: v for k, v in hp.items() if k in ("scale", "leak")}
        if "mem" in hp:
            kw["n_memristors"] = hp["mem"]
        return PhotonicMemristiveReservoir(
            data_label,
            seq_length,
            n_modes=(num_qubits if num_qubits else 6),
            n_photons=(hidden_size if hidden_size else 3),
            random_id=random_id,
            **kw,
        )
    # Capacity control for the memristive reservoir: identical sequential
    # readout, no optical memory (see lib/reservoir.py).
    if model_name in ("photonic_seqreservoir", "seqreservoir"):
        from .reservoir import PhotonicMemristiveReservoir, parse_reservoir_ansatz

        hp = parse_reservoir_ansatz(ansatz)
        return PhotonicMemristiveReservoir(
            data_label,
            seq_length,
            n_modes=(num_qubits if num_qubits else 6),
            n_photons=(hidden_size if hidden_size else 3),
            random_id=random_id,
            n_memristors=0,
            **{k: v for k, v in hp.items() if k == "scale"},
        )
    raise ValueError(f"Model {model_name!r} is not supported.")
