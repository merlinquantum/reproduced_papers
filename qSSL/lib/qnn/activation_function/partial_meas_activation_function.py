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

import logging
from math import floor

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .activation_function import ActivationFunction


class PartialMeasActivationFunction(ActivationFunction):
    """
    Creates activation function circuit in which a subset of qubits are measured and set to measurement value. Qubits to
     be to be measured will be equally spaced in circuit.
    """

    def __init__(self, n_measurements):
        super().__init__()
        self.n_measurements = n_measurements

    def get_quantum_circuit(self, n_qubits):
        if self.n_measurements == "half":
            self.n_measurements = floor(n_qubits / 2)

        if self.n_measurements > n_qubits:
            raise ValueError(
                "Activation function was asked to measure more qubits than exist in the circuit."
            )

        if n_qubits % self.n_measurements != 0:
            logging.warning(
                f"In acivation function, number of qubits ({n_qubits}) is not multiple of number of measurements "
                f"({self.n_measurements}), measurements will not be equally spaced in circuit."
            )

        self.qr = QuantumRegister(n_qubits, name="qr")
        self.cr = ClassicalRegister(self.n_measurements, name="activation_cr")
        self.qc = QuantumCircuit(self.qr, self.cr, name="Partial measurement")

        step = floor(n_qubits / self.n_measurements)

        for i, qubit in enumerate([step * j for j in range(0, self.n_measurements)]):
            self.qc.measure(self.qr[qubit], self.cr[i])

        return self.qc
