import cirq
import numpy as np

from .utils import get_angles, is_power_of_two


class RBS(cirq.testing.TwoQubitGate):
    display_angle = False

    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def _unitary_(self):
        a = np.cos(self.theta)
        b = np.sin(self.theta)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, a, b, 0],
                [0, -b, a, 0],
                [0, 0, 0, 1],
            ]
        )

    def _circuit_diagram_info_(self, args):
        if self.display_angle:
            t = args.format_radians(self.theta)
            return f"B({t})", f"S({t})"
        else:
            return "B", "S"

    def __repr__(self):
        if self.display_angle:
            from cirq._compat import proper_repr

            t = proper_repr(self.theta)
            return f"RBSGate(theta={t})"
        else:
            return "RBSGate"

    def __pow__(self, power) -> "RBS":
        return RBS(cirq.mul(self.theta, power))


class VectorLoader(cirq.Gate):
    def __init__(self, vector, x_flip=True):
        super().__init__()
        vector = np.array(vector).squeeze()
        assert is_power_of_two(len(vector)), "vector size is not power of two"
        self.vector = vector
        self.x_flip = x_flip
        self.thetas, self.wires = get_angles(vector)

    def _num_qubits_(self):
        return len(self.vector)

    def _decompose_(self, qubits):
        if self.x_flip:
            yield cirq.X(qubits[0])
        for theta, (i, j) in zip(self.thetas, self.wires):
            yield RBS(theta)(qubits[i], qubits[j])

    def _circuit_diagram_info_(self, args):
        return [f"VectorLoader({q})" for q in range(self._num_qubits_())]


class MatrixLoader(cirq.Gate):
    def __init__(self, matrix, x_flip=True):
        super().__init__()
        matrix = np.array(matrix)
        assert len(matrix) == len(matrix[0]), "matrix is not square"
        assert is_power_of_two(len(matrix)), "matrix size is not power of two"
        self.matrix = matrix
        self.x_flip = x_flip
        self.thetas_rows, self.wires_rows = get_angles(np.linalg.norm(matrix, axis=1))
        self.thetas_columns, self.wires_columns = get_angles(matrix)

    def _num_qubits_(self):
        return len(self.matrix) * 2

    def _decompose_(self, qubits):
        d = self._num_qubits_() // 2
        if self.x_flip:
            yield cirq.X(qubits[0])
            yield cirq.X(qubits[d])
        for theta, (i, j) in zip(self.thetas_rows, self.wires_rows):
            yield RBS(theta)(qubits[i], qubits[j])
        for idx, thetas in enumerate(self.thetas_columns):
            for theta, (i, j) in zip(thetas, self.wires_columns):
                yield RBS(theta)(qubits[d + i], qubits[d + j]).controlled_by(
                    qubits[idx]
                )

    def _circuit_diagram_info_(self, args):
        return [f"MatrixLoader({q})" for q in range(self._num_qubits_())]
