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

from qiskit.circuit import Parameter

from .variational_ansatz import VariationalAnsatz


class SimCirc19(VariationalAnsatz):
    """
    A variational circuit ansatz. Prepares quantum circuit object for variational circuit 19 in arXiv:1905.10876.
    Between each layer an activation function can be applied using appropriate nonlinear activation function.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.rx(param, self.qr[i])
            self.param_counter += 1

        for i in range(0, n_data_qubits):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.rz(param, self.qr[i])
            self.param_counter += 1

        return self.qc

    def add_entangling_gates(self, n_data_qubits):
        param = Parameter(f"ansatz{str(self.param_counter)}")
        self.qc.crx(param, self.qr[n_data_qubits - 1], self.qr[0])
        self.param_counter += 1
        for i in reversed(range(1, n_data_qubits)):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.crx(param, self.qr[i - 1], self.qr[i])
            self.param_counter += 1
        return self.qc
