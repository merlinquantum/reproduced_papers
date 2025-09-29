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

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .variational_ansatz import VariationalAnsatz


class Abbas(VariationalAnsatz):
    """
    A variational circuit ansatz. Based on Figure 5 of arXiv:2011.00027 - which itself is based on circuit 15 in
    arXiv:1905.10876 but with all-to-all CNOT connectivity.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name="qr")
        self.qc = QuantumCircuit(self.qr, name="Shifted circ")

        for layer_no in range(self.layers):
            if layer_no == 0:
                self.add_rotations(n_data_qubits)
            for _sweep in range(0, self.sweeps_per_layer):
                self.add_entangling_gates(n_data_qubits)
                self.add_rotations(n_data_qubits)
            if layer_no < self.layers - 1:
                self.apply_activation_function(n_data_qubits)
        return self.qc

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.ry(param, self.qr[i])
            self.param_counter += 1
        return self.qc

    def add_entangling_gates(self, n_data_qubits):
        for i in range(n_data_qubits):
            for j in range(i + 1, n_data_qubits):
                self.qc.cx(self.qr[i], self.qr[j])
        return self.qc
