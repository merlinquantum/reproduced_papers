import merlin as ml
import numpy as np
import perceval as pcvl
import perceval.components as comp
import torch
from lib.feedback import FeedbackLayer, FeedbackLayerNARMA
from lib.training import R_to_theta, encode_phase


class QuantumReservoirFeedback(torch.nn.Module):
    """
    A Quantum Reservoir Computing model with a feedback loop mechanism.
    This model uses a photonic quantum circuit where one MerLin input encodes the data and another MerLin input encodes
    the feedback state from the previous time step (mimicking the memristive effect).
    """

    def __init__(self, input_dim: int = 1, n_modes: int = 3, memory: int = 5):
        """
        Initialize the Quantum Reservoir with Feedback.

        Args:
            input_dim (int, optional): Dimension of the input data. Defaults to 1.
            n_modes (int, optional): Number of modes (qumodes) in the photonic circuit. Defaults to 3.
            memory (int, optional): Size/depth of the feedback memory buffer. Defaults to 5.
        """
        super().__init__()

        self.n_modes = n_modes

        # Create circuit that accepts TWO inputs: px_0 (data) and px_1 (feedback)
        circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim + 1,
            circuit=circuit,
            trainable_parameters=["theta"],
            input_parameters=["px"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True,
        )

        self.feedback = FeedbackLayer(memory_size=memory)

    def gen_circuit(self, n_modes: int) -> pcvl.Circuit:
        """
        Generates the Perceval quantum circuit structure for the reservoir.
        The circuit includes:
        1. An encoding section for the input data (px_0).
        2. A memory/feedback section for the feedback signal (px_1).
        3. A measurement/mixing section with trainable parameters (theta).

        Args:
            n_modes (int): Number of modes in the circuit.

        Returns:
            pcvl.Circuit: The constructed quantum circuit.
        """
        # Encoding with input data (px_0)
        phi_enc = pcvl.P(f"px_{0}")
        u_enc_u_1 = (
            pcvl.Circuit(2, name="U_enc, U_1")
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=phi_enc))
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=pcvl.P(f"theta_{1}")))
        )

        # Memory/feedback circuit with feedback input (px_1)
        u_mem = (
            pcvl.Circuit(2, name="U_mem")
            .add((0, 1), comp.BS())
            .add(0, comp.PS(phi=pcvl.P(f"px_{1}")))  # FEEDBACK INPUT
            .add((0, 1), comp.BS())
        )

        # Measurement circuit
        u_2 = (
            pcvl.Circuit(2, name="U_2")
            .add(1, comp.PS(phi=pcvl.P(f"theta_{2}")))
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=pcvl.P(f"theta_{3}")))
            .add((0, 1), comp.BS())
        )

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc_u_1)
        quantum_circ.add(1, u_mem)
        quantum_circ.add(0, u_2)

        return quantum_circ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model for a single time step or a batch of independent inputs.
        Note: This method handles the feedback state internal update.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output probability distribution from the quantum layer.
        """
        batch_size = x.shape[0]

        # If we have no R yet, start from R=0.5 (balanced) like typical initialization
        if self.feedback.last_feedback is None:
            R_t = torch.full((batch_size, 1), 0.5, device=x.device, dtype=x.dtype)
        else:
            R_scalar = self.feedback.last_feedback
            R_t = R_scalar.view(1, 1).expand(batch_size, 1)

        # compute the encoding phase
        phi_enc = encode_phase(x)

        # compute theta from R_t
        theta_t = R_to_theta(R_t)

        # pass phi_enc and theta_t into the quantum layer as px_0 and px_1
        quantum_input = torch.cat([phi_enc, theta_t], dim=1)
        out = self.quantum_layer(quantum_input)

        # update R for next step (using current output)
        _ = self.feedback(out)

        return out

    def reset_feedback(self) -> None:
        """
        Resets the internal feedback memory state.
        Should be called at the start of each new sequence or epoch.
        """
        self.feedback.reset()


class QuantumReservoirFeedbackTimeSeries(torch.nn.Module):
    """
    A Quantum Reservoir Computing model optimized for Time Series tasks (e.g., NARMA).
    It processes an entire sequence in a loop within the forward pass, maintaining the feedback state across time steps.
    """

    def __init__(self, input_dim: int = 1, n_modes: int = 3, memory: int = 5):
        """
        Initialize the Time Series Quantum Reservoir.

        Args:
            input_dim (int, optional): Dimension of the input data. Defaults to 1.
            n_modes (int, optional): Number of modes in the circuit. Defaults to 3.
            memory (int, optional): Memory depth for the feedback calculation. Defaults to 5.
        """
        super().__init__()
        self.n_modes = n_modes
        self.circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim + 1,  # input + feedback
            circuit=self.circuit,
            trainable_parameters=[],  # Fixed reservoir
            input_parameters=["px_0", "px_1"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True,
        )

        self.feedback = FeedbackLayerNARMA(memory_size=memory)

    def gen_circuit(self, n_modes: int) -> pcvl.Circuit:
        """
        Generates the Perceval quantum circuit structure.

        Args:
            n_modes (int): Number of modes.

        Returns:
            pcvl.Circuit: The constructed quantum circuit.
        """
        # Encoding with input data (px_0)
        phi_enc = pcvl.P(f"px_{0}")
        u_enc = (
            pcvl.Circuit(2, name="U_enc")
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=phi_enc))
            .add((0, 1), comp.BS())
        )

        # Memory/feedback circuit with feedback input (px_1)
        u_mem = (
            pcvl.Circuit(2, name="U_mem")
            .add((0, 1), comp.BS())
            .add(0, comp.PS(phi=pcvl.P(f"px_{1}")))
            .add((0, 1), comp.BS())
        )

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc)
        quantum_circ.add(1, u_mem)

        return quantum_circ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an entire time series sequence.
        Iterates through the sequence dimension, updating the reservoir state step-by-step using the feedback mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
                              or (sequence_length, input_dim).

        Returns:
            torch.Tensor: The sequence of output states, shape (batch_size, sequence_length, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape
        outputs = []

        # Reset feedback at the start of the sequence
        self.reset_feedback()

        for t in range(seq_len):
            # 1. Get current input step
            x_t = x[:, t, :]  # (Batch, 1)

            # 2. Get current feedback state R_t
            R_scalar = self.feedback.get_R(device=x.device, dtype=x.dtype)

            # If batch > 1, ensure R_scalar matches batch dim.
            if R_scalar.dim() == 0:
                R_t = R_scalar.view(1, 1).expand(batch_size, 1)
            else:
                R_t = R_scalar.view(batch_size, 1)

            # 3. Encode Inputs
            phi_enc = encode_phase(x_t)
            theta_t = R_to_theta(R_t)

            # 4. Step the Quantum Layer
            step_input = torch.cat([phi_enc, theta_t], dim=1)
            out_t = self.quantum_layer(step_input)

            # 5. Update Feedback for next step
            _ = self.feedback(out_t)

            outputs.append(out_t)

        return torch.stack(outputs, dim=1)

    def reset_feedback(self) -> None:
        """
        Resets the internal feedback memory.
        """
        self.feedback.reset()


class QuantumReservoirNoMem(torch.nn.Module):
    """
    A baseline Quantum Reservoir model WITHOUT memory/feedback.
    Uses a standard feedforward quantum circuit where the "memory" component is replaced by a static or trainable phase
    shift.
    """

    def __init__(self, input_dim: int = 1, n_modes: int = 3):
        """
        Initialize the No-Memory Quantum Reservoir.

        Args:
            input_dim (int, optional): Dimension of the input data. Defaults to 1.
            n_modes (int, optional): Number of modes in the circuit. Defaults to 3.
        """
        super().__init__()
        self.n_modes = n_modes

        circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim,
            circuit=circuit,
            trainable_parameters=["theta"],
            input_parameters=["px"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True,
        )

    def gen_circuit(self, n_modes: int) -> pcvl.Circuit:
        """
        Generates the circuit for the no-memory baseline.

        Args:
            n_modes (int): Number of modes.

        Returns:
            pcvl.Circuit: The constructed quantum circuit.
        """
        # Encoding with input data (px_0)
        phi_enc = pcvl.P("px_0")
        u_enc_u_1 = (
            pcvl.Circuit(2, name="U_enc, U_1")
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=phi_enc))
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=pcvl.P("theta_1")))
        )

        # Trainable angle instead of memristor
        u_mem = (
            pcvl.Circuit(2, name="U_mem_static")
            .add((0, 1), comp.BS())
            # .add(0, comp.PS(phi=pcvl.P("theta_2")))
            .add(0, comp.PS(phi=np.pi / 2))
            .add((0, 1), comp.BS())
        )

        # Measurement circuit
        u_2 = (
            pcvl.Circuit(2, name="U_2")
            .add(1, comp.PS(phi=pcvl.P("theta_3")))
            .add((0, 1), comp.BS())
            .add(1, comp.PS(phi=pcvl.P("theta_4")))
            .add((0, 1), comp.BS())
        )

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc_u_1)
        quantum_circ.add(1, u_mem)
        quantum_circ.add(0, u_2)

        return quantum_circ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the no-memory model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            torch.Tensor: Output probability distribution.
        """
        return self.quantum_layer(x)
