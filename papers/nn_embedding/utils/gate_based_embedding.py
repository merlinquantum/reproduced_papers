"""Gate-based quantum embedding circuits.

Adapted from the original repository:
https://github.com/takh04/neural-quantum-embedding
"""

import pennylane as qml
from pennylane import numpy as np


# exp(ixZ) gate
def exp_Z(x, wires, inverse=False):
    if not inverse:
        qml.RZ(-2 * x, wires=wires)
    elif inverse:
        qml.RZ(2 * x, wires=wires)


# exp(ixZZ) gate
def exp_ZZ1(x, wires, inverse=False):
    if not inverse:
        qml.CNOT(wires=wires)
        qml.RZ(-2 * x, wires=wires[1])
        qml.CNOT(wires=wires)
    elif inverse:
        qml.CNOT(wires=wires)
        qml.RZ(2 * x, wires=wires[1])
        qml.CNOT(wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires, inverse=False):
    if not inverse:
        qml.CNOT(wires=wires)
        qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
        qml.CNOT(wires=wires)
    elif inverse:
        qml.CNOT(wires=wires)
        qml.RZ(2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
        qml.CNOT(wires=wires)


class EmbeddingCallable:
    def __init__(self, N_layers: int = 3):
        self.N_layers = N_layers

    # Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
    def QuantumEmbedding1(self, input):
        for _ in range(self.N_layers):
            for j in range(8):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(7):
                exp_ZZ2(input[..., k], input[..., k + 1], wires=[k, k + 1])
            exp_ZZ2(input[..., 7], input[..., 0], wires=[7, 0])

    def QuantumEmbedding1Trainable(self, input, params):
        param_index = 0
        for _ in range(self.N_layers):
            for j in range(8):
                qml.RY(params[param_index], wires=j)
                param_index += 1
            for j in range(8):
                for k in range(j + 1, 8):
                    qml.adjoint(qml.S(wires=j))
                    qml.adjoint(qml.S(wires=k))
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.RZ(params[param_index], wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.S(wires=j)
                    qml.S(wires=k)

                    param_index += 1
            for j in range(8):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(7):
                exp_ZZ2(input[..., k], input[..., k + 1], wires=[k, k + 1])
            exp_ZZ2(input[..., 7], input[..., 0], wires=[7, 0])

    # Quantum Embedding 2 for model 2
    def QuantumEmbedding2(self, input):
        for _ in range(self.N_layers):
            for j in range(8):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(7):
                exp_ZZ1(input[..., 8 + k], wires=[k, k + 1])
            exp_ZZ1(input[..., 15], wires=[7, 0])

    def QuantumEmbedding2Trainable(self, input, params):
        param_index = 0
        for _ in range(self.N_layers):
            for j in range(8):
                qml.RY(params[param_index], wires=j)
                param_index += 1
            for j in range(8):
                for k in range(j + 1, 8):
                    qml.adjoint(qml.S(wires=j))
                    qml.adjoint(qml.S(wires=k))
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.RZ(params[param_index], wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.S(wires=j)
                    qml.S(wires=k)

                    param_index += 1
            for j in range(8):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(7):
                exp_ZZ1(input[..., 8 + k], wires=[k, k + 1])
            exp_ZZ1(input[..., 15], wires=[7, 0])

    # Add 4 qubit embedding for demonstrations
    def Four_QuantumEmbedding1(self, input):
        for _ in range(self.N_layers):
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(3):
                exp_ZZ2(input[..., k], input[..., k + 1], wires=[k, k + 1])
            exp_ZZ2(input[..., 3], input[..., 0], wires=[3, 0])

    def Four_QuantumEmbedding1Trainable(self, input, params):
        param_index = 0
        for _ in range(self.N_layers):
            for j in range(4):
                qml.RY(params[param_index], wires=j)
                param_index += 1
            for j in range(4):
                for k in range(j + 1, 4):
                    qml.adjoint(qml.S(wires=j))
                    qml.adjoint(qml.S(wires=k))
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.RZ(params[param_index], wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.S(wires=j)
                    qml.S(wires=k)

                    param_index += 1
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(3):
                exp_ZZ2(input[..., k], input[..., k + 1], wires=[k, k + 1])
            exp_ZZ2(input[..., 3], input[..., 0], wires=[3, 0])

    def Four_QuantumEmbedding2(self, input):
        for _ in range(self.N_layers):
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(3):
                exp_ZZ1(input[..., 4 + k], wires=[k, k + 1])
            exp_ZZ1(input[..., 7], wires=[3, 0])

    def Four_QuantumEmbedding2Trainable(self, input, params):
        param_index = 0
        for _ in range(self.N_layers):
            for j in range(4):
                qml.RY(params[param_index], wires=j)
                param_index += 1
            for j in range(4):
                for k in range(j + 1, 4):
                    qml.adjoint(qml.S(wires=j))
                    qml.adjoint(qml.S(wires=k))
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.RZ(params[param_index], wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.Hadamard(wires=j)
                    qml.Hadamard(wires=k)
                    qml.S(wires=j)
                    qml.S(wires=k)

                    param_index += 1
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[..., j], wires=j)
            for k in range(3):
                exp_ZZ1(input[..., 4 + k], wires=[k, k + 1])
            exp_ZZ1(input[..., 7], wires=[3, 0])


# Add 4 qubit noisy embedding for demonstrations
def U_SU4(params, wires):  # 15 params
    qml.U3(params[..., 0], params[..., 1], params[..., 2], wires=wires[0])
    qml.U3(params[..., 3], params[..., 4], params[..., 5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[..., 6], wires=wires[0])
    qml.RZ(params[..., 7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[..., 8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[..., 9], params[..., 10], params[..., 11], wires=wires[0])
    qml.U3(params[..., 12], params[..., 13], params[..., 14], wires=wires[1])


def U_TTN(params, wires):  # 2 params
    qml.RY(params[..., 0], wires=wires[0])
    qml.RY(params[..., 1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def FourQCNN(input):
    U = U_SU4
    num_params = 15

    param1 = input[..., 0:num_params]
    param2 = input[..., num_params : 2 * num_params]
    U(param1, wires=[0, 1])
    U(param1, wires=[2, 3])
    U(param1, wires=[1, 2])
    U(param1, wires=[3, 0])
    U(param2, wires=[0, 2])


def QCNN(input):
    # if ansatz == "SU4":
    U = U_SU4
    num_params = 15
    # elif ansatz == 'TTN':
    #   U = U_TTN
    #   num_params = 2

    param1 = input[..., 0:num_params]
    param2 = input[..., num_params : 2 * num_params]
    param3 = input[..., 2 * num_params : 3 * num_params]

    first_layer_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    second_layer_pairs = [(1, 2), (3, 4), (5, 6), (7, 0)]
    final_layer_pairs = [(0, 2), (4, 6)]

    for wires in first_layer_pairs:
        U(param1, wires=list(wires))

    for wires in second_layer_pairs:
        U(param1, wires=list(wires))

    for wires in final_layer_pairs:
        U(param2, wires=list(wires))

    U(param3, wires=[0, 4])
