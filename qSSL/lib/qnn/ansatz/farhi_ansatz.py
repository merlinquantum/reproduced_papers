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


class FarhiAnsatz(VariationalAnsatz):
    """
    A variational circuit ansatz. Prepares quantum circuit object for variational circuit consisting of X⊗X and Z⊗X
    rotations between n-1 qubits and single qubit described in arXiv:1802.06002.
    Between each layer (combined X⊗X and Z⊗X rotations), activation function can be applied using appropriate
    nonlinear activation function.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        """
        :param layers: Number of layers for ansatz. Between each layer the activation function is applied.
        :param sweeps_per_layer: Number parameterised gate sweeps in a single layer. No non-linearity will be applied
        between sweeps.
        :param activation_function: Type of activation function to be applied between each layer. Allowed values are
        'partial_measurement' and 'null'.
        :param measurement_spacing: For 'partial measurement' activation function type, qubits to be measured will be
        uniformly spaced along quantum register by this amount.
        """
        super().__init__(layers, sweeps_per_layer, activation_function)

    def add_entangling_gates(self, n_data_qubits):
        self.rxx_to_all(n_data_qubits)
        self.rzx_to_all(n_data_qubits)

    def add_rotations(self, n_data_qubits):
        pass

    def rxx_to_all(self, n_data_qubits):
        for i in range(n_data_qubits - 1):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.rxx(param, self.qr[-1], self.qr[i])
            self.param_counter += 1
        return self.qc

    def rzx_to_all(self, n_data_qubits):
        for i in range(n_data_qubits - 1):
            param = Parameter(f"ansatz{str(self.param_counter)}")
            self.qc.rzx(param, self.qr[-1], self.qr[i])
            self.param_counter += 1
        return self.qc
