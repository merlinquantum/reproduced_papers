import cirq


class GateNoise(cirq.PointOptimizer):
    def __init__(self, error_rate=0.005):
        super().__init__()
        self.error_rate = error_rate

    def optimization_at(self, circuit, index, op):
        if isinstance(op.gate, cirq.Gate) and not isinstance(
                op.gate, cirq.MeasurementGate):
            noise_gate = cirq.depolarize(self.error_rate).on_each(*op.qubits)
            decomposition = [op.gate(*op.qubits)] + noise_gate
            return cirq.PointOptimizationSummary(clear_span=1,
                                                 clear_qubits=op.qubits,
                                                 new_operations=decomposition)
        else:
            return None

    def apply(self, circuit):
        self.optimize_circuit(circuit)
        return circuit


class MeasurementNoise(cirq.PointOptimizer):
    def __init__(self, error_rate=0.005):
        super().__init__()
        self.error_rate = error_rate

    def optimization_at(self, circuit, index, op):
        if isinstance(op.gate, cirq.MeasurementGate):
            noise_gate = cirq.bit_flip(self.error_rate).on_each(*op.qubits)
            decomposition = [op.gate(*op.qubits)] + noise_gate
            return cirq.PointOptimizationSummary(clear_span=1,
                                                 clear_qubits=op.qubits,
                                                 new_operations=decomposition)
        else:
            return None

    def apply(self, circuit):
        self.optimize_circuit(circuit)
        return circuit
