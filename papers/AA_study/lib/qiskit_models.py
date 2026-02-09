import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_ROOT = Path(__file__).resolve().parent
TORCHQUANTUM_ROOT = LIB_ROOT / "torchquantum"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TORCHQUANTUM_ROOT))


import torchquantum as tq
import torchquantum.functional as tqf


class single_qubit_model(tq.QuantumModule):
    class SingleQubitModelQLayer(tq.QuantumModule):
        def __init__(self, num_layers: int = 10):
            super().__init__()
            self.n_wires = 1

            # gates with trainable parameters
            self.gates = tq.QuantumModuleList(
                [
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                ]
                * num_layers
            )

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            Apply the parameterized single-qubit gate sequence.

            Notes
            -----
            When using static mode or exporting to Qiskit:
            1) Add `@tq.static_support` before `forward`.
            2) Pass `static=self.static_mode` and `parent_graph=self.graph`
               to all `tqf` functional calls.
            """
            self.q_device = q_device

            # some trainable gates (instantiated ahead of time)
            for gates in self.gates:
                gates(self.q_device, wires=0)

    def __init__(self, num_layers: int = 10, return_probs: bool = True):
        """
        Build a single-qubit TorchQuantum (Qiskit) model.

        Parameters
        ----------
        num_layers : int, optional
            Number of repeated gate blocks.
        return_probs : bool, optional
            Whether `forward` returns probabilities instead of logits.
        """
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.SingleQubitModelQLayer(num_layers=num_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.return_probs = return_probs

    def forward(self, x):
        """
        Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            If `return_probs=True`, probabilities of shape (N, 2);
            otherwise logits of shape (N, 2).
        """
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # Convert single expectation value into 2-class logits
        logits = torch.cat([x, -x], dim=1)
        if self.return_probs:
            return torch.softmax(logits, dim=1)
        return logits


# TODO Add a num_classes parameter
class qiskit_QCNN(tq.QuantumModule):
    class QiskitQCNNQLayer(tq.QuantumModule):
        def __init__(self, num_qubits: int = 10):
            """
            Build the QCNN quantum layer.

            Parameters
            ----------
            num_qubits : int, optional
                Number of qubits/wires in the circuit.
            """
            super().__init__()

            num_u_gates = 0
            num_v_gates = 0
            num_qubits_alive = num_qubits

            while num_qubits_alive > 1:
                num_u_gates += num_qubits_alive - 1
                num_v_gates += num_qubits_alive // 2
                num_qubits_alive = num_qubits_alive - (num_qubits_alive // 2)

            self.n_wires = num_qubits

            # gates with trainable parameters
            self.U_gates = tq.QuantumModuleList(
                [
                    tq.RXX(has_params=True, trainable=True),
                    tq.RYY(has_params=True, trainable=True),
                    tq.RZZ(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.RX(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                ]
                * num_u_gates
            )
            self.V_gates = tq.QuantumModuleList(
                [tq.CRX(has_params=True, trainable=True)] * num_v_gates
            )

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            Apply the QCNN layer to the quantum device.

            Notes
            -----
            When using static mode or exporting to Qiskit:
            1) Add `@tq.static_support` before `forward`.
            2) Pass `static=self.static_mode` and `parent_graph=self.graph`
               to all `tqf` functional calls.
            """
            self.q_device = q_device

            u_gate_index = 0
            v_gate_index = 0
            qubits_alive = [i for i in range(self.n_wires)]

            while len(qubits_alive) > 1:
                for i in range(0, len(qubits_alive) - 1, 2):
                    for local_idx, gate_to_apply in enumerate(
                        self.U_gates[u_gate_index : u_gate_index + 9]
                    ):
                        if local_idx < 3:
                            gate_to_apply(
                                self.q_device,
                                wires=[qubits_alive[i], qubits_alive[i + 1]],
                            )
                        elif local_idx < 6:
                            gate_to_apply(self.q_device, wires=qubits_alive[i])
                        else:
                            gate_to_apply(self.q_device, wires=qubits_alive[i + 1])
                    u_gate_index += 9
                for i in range(1, len(qubits_alive) - 1, 2):
                    for local_idx, gate_to_apply in enumerate(
                        self.U_gates[u_gate_index : u_gate_index + 9]
                    ):
                        if local_idx < 3:
                            gate_to_apply(
                                self.q_device,
                                wires=[qubits_alive[i], qubits_alive[i + 1]],
                            )
                        elif local_idx < 6:
                            gate_to_apply(self.q_device, wires=qubits_alive[i])
                        else:
                            gate_to_apply(self.q_device, wires=qubits_alive[i + 1])
                    u_gate_index += 9

                for i in range(1, len(qubits_alive), 2):
                    self.V_gates[v_gate_index](
                        self.q_device, wires=[qubits_alive[i], qubits_alive[i - 1]]
                    )
                    v_gate_index += 1

                qubits_alive = qubits_alive[::2]

    def __init__(self, num_qubits: int = 10, return_probs: bool = True):
        """
        Build the (Qiskit) QCNN model.

        Parameters
        ----------
        num_qubits : int, optional
            Number of qubits/wires in the circuit.
        return_probs : bool, optional
            Whether `forward` returns probabilities instead of logits.
        """
        super().__init__()
        self.n_wires = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QiskitQCNNQLayer(num_qubits=num_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.return_probs = return_probs

    def _preprocess_input(self, x):
        """
        Flatten and truncate inputs to match the circuit width.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (N, <= 2**n_wires).
        """
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        max_dim = 2**self.n_wires
        if x.shape[1] > max_dim:
            x = x[:, :max_dim]
        return x

    def _first_qubit_probs(self, x):
        """
        Measure all qubits and return probabilities of qubit 0.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input tensor.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (N, 2) for the first qubit.
        """
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = x[:, 0]
        return torch.stack([(1.0 + z) / 2.0, (1.0 - z) / 2.0], dim=1)

    def forward(self, x):
        """
        Run a forward pass returning first-qubit probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (N, 2).
        """
        x = self._preprocess_input(x)
        return self._first_qubit_probs(x)

    def forward_logits(self, x):
        """
        Run a forward pass returning logits derived from the first qubit.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logits of shape (N, 2).
        """
        x = self._preprocess_input(x)
        probs = self._first_qubit_probs(x)
        z = probs[:, 0] - probs[:, 1]
        return torch.stack([z, -z], dim=1)
