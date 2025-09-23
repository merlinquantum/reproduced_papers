# MIT License
#
# Copyright (c) 2021 Ben Jaderberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.extensions import CXGate, U1Gate, U2Gate, U3Gate


class RZXGate(Gate):
    """Two-qubit XZ-rotation gate. Modified XX-rotation gate from Qiskit.

    This gate corresponds to the rotation U(θ) = exp(-1j * θ * Z⊗X / 2)
    """

    def __init__(self, theta):
        """Create new rzx gate."""
        super().__init__("rzx", 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""

        definition = []
        q = QuantumRegister(2, "q")
        theta = self.params[0]
        rule = [
            (U3Gate(np.pi / 2, theta, 0), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U2Gate(-np.pi, np.pi - theta), [q[0]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return RZXGate(-self.params[0])


def rzx(self, theta, qubit1, qubit2):
    """Apply RZX to circuit."""
    return self.append(RZXGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.rzx = rzx
