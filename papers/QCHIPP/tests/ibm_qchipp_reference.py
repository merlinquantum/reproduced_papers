# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: the nine circuit-construction functions below were extracted
# unchanged from IBM/Q-CHIPP v1.0.0 (commit
# 9f72bd9fc87522bc45c806f9849d07210a64e48d), files libraries/qcnn_lib.py and
# qrac/qrac_parametervector.py, and reformatted with Ruff. They serve only as an
# independent Qiskit oracle for tests/test_gate.py.
"""IBM Q-CHIPP v1.0.0 reference circuits used as a test oracle."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import SwapGate


def conv_circuit(params, conv_ansatz=0):
    if conv_ansatz == 0:
        # two qubit unitary N as defined in Vatan and Williams
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)

    elif conv_ansatz == 1:
        # full KAK decomposition
        assert len(params) == 15
        target = QuantumCircuit(2)
        target.rz(params[0], 0)
        target.rz(params[1], 1)
        target.ry(params[2], 0)
        target.ry(params[3], 1)
        target.rz(params[4], 0)
        target.rz(params[5], 1)
        # target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[6], 0)
        target.ry(params[7], 1)
        target.cx(0, 1)
        target.ry(params[8], 1)
        target.cx(1, 0)
        # target.rz(np.pi / 2, 0)
        target.rz(params[9], 0)
        target.rz(params[10], 1)
        target.ry(params[11], 0)
        target.ry(params[12], 1)
        target.rz(params[13], 0)
        target.rz(params[14], 1)

    else:
        raise AssertionError("Invalid Ansatz number.")

    return target


def conv_layer(num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + num_param)], conv_ansatz),
            [q1, q2],
        )
        if test:
            qc.barrier()
        param_index += num_param

    if periodic:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(
                conv_circuit(
                    params[param_index : (param_index + num_param)], conv_ansatz
                ),
                [q1, q2],
            )
            if test:
                qc.barrier()
            param_index += num_param
    else:
        for q1, q2 in zip(qubits[1::2], qubits[2::2]):  # no periodic connectivity
            qc = qc.compose(
                conv_circuit(
                    params[param_index : (param_index + num_param)], conv_ansatz
                ),
                [q1, q2],
            )
            if test:
                qc.barrier()
            param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def conv_layer_first(
    num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False
):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer First")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + num_param)], conv_ansatz),
            [q1, q2],
        )
        if test:
            qc.barrier()
        param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def conv_layer_second(
    num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False
):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer Second")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    if periodic:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(
                conv_circuit(
                    params[param_index : (param_index + num_param)], conv_ansatz
                ),
                [q1, q2],
            )
            if test:
                qc.barrier()
            param_index += num_param
    else:
        for q1, q2 in zip(qubits[1::2], qubits[2::2]):  # no periodic connectivity
            qc = qc.compose(
                conv_circuit(
                    params[param_index : (param_index + num_param)], conv_ansatz
                ),
                [q1, q2],
            )
            if test:
                qc.barrier()
            param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params, pool_ansatz=0):
    if pool_ansatz == 0:
        # Ansatz from the qiskit tutorial
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

    elif pool_ansatz == 1:
        # Non-parametric dynamic circuit (Tanvi's version)
        assert len(params) == 0
        target = QuantumCircuit(2, 1)
        target.cx(1, 0)
        target.h(0)
        target.measure(
            0, 0
        )  # measure qubit 0 and store it's classical bit value in cbit[0]
        with target.if_test(
            (0, 1)
        ):  # if measured state of cbit[0] is true, flip qubit 1
            target.x(1)

    elif pool_ansatz == 2:
        # From "Quantum convolutional neural network for classical data classification"
        assert len(params) == 2
        target = QuantumCircuit(2)
        target.crz(params[0], 0, 1)
        target.x(0)
        target.crx(params[1], 0, 1)

    elif pool_ansatz == 3:
        # Similar to controlled-U3 rotation
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(params[0], 1)
        target.cx(0, 1)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.rz(params[2], 1)

    else:
        raise AssertionError("Invalid Ansatz number.")

    return target


def pool_layer(sources, sinks, param_prefix, pool_ansatz=0, test=False):
    dynamic_circuit = [
        False,
        True,
        False,
        False,
    ]  # in order corresponding to each different type of pool ansatz
    dc = dynamic_circuit[pool_ansatz]
    num_params = [
        3,
        0,
        2,
        3,
    ]  # in order corresponding to each different type of pool ansatz
    num_param = num_params[pool_ansatz]
    if dc:
        num_qubits = len(sources) + len(sinks)
        num_clbits = len(sources)
        qc = QuantumCircuit(num_qubits, num_clbits, name="Pooling Layer")
        params = []
        clbit_index = 0
        for source, sink in zip(sources, sinks):
            qc.compose(
                pool_circuit(params, pool_ansatz),
                qubits=[source, sink],
                clbits=[clbit_index],
                inplace=True,
            )
            clbit_index += 1
            if test:
                qc.barrier()
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits, num_clbits)
        qc.append(qc_inst, range(num_qubits), range(num_clbits))

    else:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * num_param)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                pool_circuit(
                    params[param_index : (param_index + num_param)], pool_ansatz
                ),
                [source, sink],
            )
            if test:
                qc.barrier()
            param_index += num_param
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))

    return qc


def swap_pyramid(num_qubits):
    assert num_qubits % 2 == 0, "Even number of qubits required"
    qc = QuantumCircuit(num_qubits, name="Swap Pyramid")
    initial_qubits = list(range(num_qubits))

    qubits = list(range(num_qubits))
    while len(qubits) > 0:
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(SwapGate(), [q1, q2])
        qubits = qubits[1:-1]

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, initial_qubits)
    return qc


def create_pyramid_ansatz(
    ansatz, num_qubits, pool_until=1, pool_circuit=0, conv_circuit=0, barrier=False
):

    # number and size of convolutional layters
    x = num_qubits
    conv_layer_sizes = []
    while x > 2:
        conv_layer_sizes.append(x)
        x = int(np.ceil(np.ceil(x / 2) / 2.0) * 2)

    center = int(num_qubits / 2)

    for layer in range(len(conv_layer_sizes)):
        nqbits = conv_layer_sizes[layer]
        if nqbits % 4 == 0:  # TODO Principle confirm number of pooling gates
            ansatz.compose(
                conv_layer(
                    num_qubits=nqbits,
                    param_prefix=f"c{layer}",
                    conv_ansatz=conv_circuit,
                ),
                list(range(center - int(nqbits / 2), center + int(nqbits / 2))),
                inplace=True,
            )
        else:
            ansatz.compose(
                conv_layer_second(num_qubits=nqbits, param_prefix=f"cs{layer}"),
                list(range(center - int(nqbits / 2), center + int(nqbits / 2))),
                inplace=True,
            )
            ansatz.compose(
                conv_layer_first(num_qubits=nqbits, param_prefix=f"cf{layer}"),
                list(range(center - int(nqbits / 2), center + int(nqbits / 2))),
                inplace=True,
            )
        if barrier:
            ansatz.barrier()

        # Pooling Layer
        midpoint = int(np.floor(np.floor(nqbits / 2) / 2.0) * 2)
        endpoint = midpoint * 2
        ansatz.compose(
            pool_layer(
                sources=list(range(0, midpoint - 1, 2))
                + list(range(midpoint + 1, endpoint, 2)),
                sinks=list(range(1, midpoint, 2))
                + list(range(midpoint, endpoint - 1, 2)),
                param_prefix=f"p{layer}",
                pool_ansatz=pool_circuit,
            ),
            list(range(center - int(endpoint / 2), center + int(endpoint / 2))),
            inplace=True,
        )
        if barrier:
            ansatz.barrier()

        # Swap pyramid
        swap_size = int(np.ceil((np.ceil(nqbits / 2) - 2) / 2.0) * 2)
        if swap_size > 0:
            ansatz.compose(
                swap_pyramid(num_qubits=swap_size),
                list(range(center - swap_size - 1, center - 1)),
                inplace=True,
            )
            ansatz.compose(
                swap_pyramid(num_qubits=swap_size),
                list(range(center + 1, center + swap_size + 1)),
                inplace=True,
            )
            if barrier:
                ansatz.barrier()

    # For the last layer with 2 qubits
    layer = layer + 1
    ansatz.compose(
        conv_layer(num_qubits=2, param_prefix=f"c{layer}"),
        list(range(center - 1, center + 1)),
        inplace=True,
    )
    if barrier:
        ansatz.barrier()

    # Pooling Layer
    ansatz.compose(
        pool_layer(sources=[0], sinks=[1], param_prefix=f"p{layer}"),
        list(range(center - 1, center + 1)),
        inplace=True,
    )

    return ansatz


def full_circuit_21p_qrac(len_bitstring: int) -> QuantumCircuit:
    assert len_bitstring % 2 == 0, "Length of bitstring should be an even number."
    num_qubits = int(len_bitstring / 2)
    params = ParameterVector("a", length=len_bitstring)

    qr = QuantumRegister(num_qubits, "q")
    circ = QuantumCircuit(qr)

    for i in range(num_qubits):
        param0 = params[2 * i]
        param1 = params[2 * i + 1]
        theta = (np.pi / 4) + (np.pi / 2) * (param1 - param0) + np.pi * param1 * param0
        circ.ry(theta, qr[i])

    return circ
