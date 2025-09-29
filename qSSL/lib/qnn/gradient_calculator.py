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


import logging

import numpy as np
from qiskit import transpile


def calculate_gradient_list(
    qnn, parameter_list, method="parameter shift", shots=100, eps=None
):
    parameter_list = np.array(parameter_list, dtype=float)

    if method == "parameter shift":
        r = 0.5  # for Farhi ansatz, e0 = -1, e1 = +1, a = 1 => r = 0.5 (Using notation in arXiv:1905.13311)

        qc_plus_list, qc_minus_list = get_parameter_shift_circuits(
            qnn, parameter_list, r
        )

        expectation_minus, expectation_plus = evaluate_gradient_jobs(
            qc_minus_list, qc_plus_list, qnn, shots
        )

        gradient_list = r * (expectation_plus - expectation_minus)

    elif method == "finite difference":
        qc_plus_list, qc_minus_list = get_finite_difference_circuits(
            qnn, parameter_list, eps
        )

        expectation_minus, expectation_plus = evaluate_gradient_jobs(
            qc_minus_list, qc_plus_list, qnn, shots
        )

        gradient_list = (expectation_plus - expectation_minus) / (2 * eps)

    else:
        raise ValueError("Invalid gradient method")

    gradient_list = gradient_list.reshape([len(parameter_list), -1])
    return gradient_list


def evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots):
    qc_minus_list = [
        transpile(circ, basis_gates=["cx", "u1", "u2", "u3"]) for circ in qc_minus_list
    ]
    qc_plus_list = [
        transpile(circ, basis_gates=["cx", "u1", "u2", "u3"]) for circ in qc_plus_list
    ]
    results = qnn.backend.run(qc_minus_list + qc_plus_list, shots=shots).result()
    expectation_plus = []
    expectation_minus = []
    num_params = len(qc_plus_list)
    for i in range(num_params):
        expectation_minus.append(qnn.get_vector_from_results(results, i))
        expectation_plus.append(qnn.get_vector_from_results(results, num_params + i))
        logging.debug(f"Gradient calculated for {i} out of {num_params} parameters")
    return np.array(expectation_minus), np.array(expectation_plus)


def get_parameter_shift_circuits(qnn, parameter_list, r):
    qc_plus_list, qc_minus_list = [], []
    for i in range(len(parameter_list)):
        shifted_params_plus = np.copy(parameter_list)
        shifted_params_plus[i] = shifted_params_plus[i] + np.pi / (4 * r)
        shifted_params_minus = np.copy(parameter_list)
        shifted_params_minus[i] = shifted_params_minus[i] - np.pi / (4 * r)

        qc_i_plus = qnn.bind_circuit(shifted_params_plus)
        qc_i_minus = qnn.bind_circuit(shifted_params_minus)
        qc_plus_list.append(qc_i_plus)
        qc_minus_list.append(qc_i_minus)

    return qc_plus_list, qc_minus_list


def get_finite_difference_circuits(qnn, parameter_list, eps):
    if type(eps) is float:
        qc_plus_list, qc_minus_list = [], []
        for i in range(len(parameter_list)):
            shifted_params_plus = np.copy(parameter_list)
            shifted_params_plus[i] = shifted_params_plus[i] + eps
            shifted_params_minus = np.copy(parameter_list)
            shifted_params_minus[i] = shifted_params_minus[i] - eps

            qc_i_plus = qnn.bind_circuit(shifted_params_plus)
            qc_i_minus = qnn.bind_circuit(shifted_params_minus)
            qc_plus_list.append(qc_i_plus)
            qc_minus_list.append(qc_i_minus)

        return qc_plus_list, qc_minus_list
    else:
        raise ValueError("eps for finite difference scheme must be float")
