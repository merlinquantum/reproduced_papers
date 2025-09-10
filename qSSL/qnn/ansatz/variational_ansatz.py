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

# This script was edited to work with qiskit 2.1.1 (Cassandre Notton)


from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, QuantumRegister


class VariationalAnsatz(ABC):
    def __init__(self, layers, sweeps_per_layer, activation_function):
        self.layers = layers
        self.sweeps_per_layer = sweeps_per_layer
        self.qr = None
        self.qc = None
        self.n_parameters_required = None
        self.activation_function = activation_function

        self.param_counter = 0

    # Base logic for creating an ansatz, can be overwritten by children
    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name="qr")
        self.qc = QuantumCircuit(self.qr, name="Shifted circ")

        for layer_no in range(self.layers):
            for _sweep in range(0, self.sweeps_per_layer):
                self.add_rotations(n_data_qubits)
                self.add_entangling_gates(n_data_qubits)
            if layer_no < self.layers - 1:
                self.apply_activation_function(n_data_qubits)
        return self.qc

    @abstractmethod
    def add_rotations(self, n_data_qubits):
        pass

    @abstractmethod
    def add_entangling_gates(self, n_data_qubits):
        pass

    def apply_activation_function(self, n_data_qubits):
        activation_function_circuit = self.activation_function.get_quantum_circuit(
            n_data_qubits
        )
        self.qc.compose(activation_function_circuit, inplace=True)
        return self.qc
