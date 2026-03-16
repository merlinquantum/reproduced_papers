import cirq
import numpy as np
import perceval as pcvl
import torch

from .gates import VectorLoader
from .noise import GateNoise, MeasurementNoise
from .utils import create_circuit, get_angles


def quantum_inner(
    x,
    y,
    repetitions=1000,
    error_rate=0.0,
    error_mitigation=True,
):
    if error_rate == 0.0:
        error_mitigation = False

    qubits = cirq.LineQubit.range(len(x))
    gates_x = cirq.decompose(VectorLoader(x)(*qubits))
    gates_y = cirq.decompose(VectorLoader(y, x_flip=False)(*qubits) ** -1)

    if error_mitigation:
        gates_m = cirq.measure(*qubits, key="all")
    else:
        gates_m = cirq.measure(qubits[0], key="m")

    circuit = cirq.Circuit(gates_x, gates_y, gates_m)

    if error_rate > 0.0:
        GateNoise(error_rate).apply(circuit)
        MeasurementNoise(error_rate).apply(circuit)
    simulator = cirq.Simulator()
    measure = simulator.run(circuit, repetitions=repetitions)

    if error_mitigation:
        hist = measure.histogram(key="all")
        # Example: for n qubits, state |1000...0> corresponds to 2^(n-1)
        total = hist.get(2 ** (len(x) - 1), 0)
        num = sum(hist.get(2**i, 0) for i in range(len(x)))
    else:
        hist = measure.histogram(key="m")
        total = hist.get(1, 0)
        num = repetitions
    inner = np.sqrt(total / num) if num > 0 else 0.0
    return inner


def quantum_inner_merlin(x, y, layer=None, shots=None, sampling_method=None):
    x_forward = torch.cat(
        (
            2 * torch.tensor(get_angles(x)[0]),
            -2 * torch.flip(torch.tensor(get_angles(y)[0]), dims=[0]),
        )
    )
    result = layer(x_forward, shots=shots, sampling_method=sampling_method).sqrt()
    if len(result.shape) == 1:
        return result[0]
    return result[0, 0]


def quantum_inner_pcvl(
    x,
    y,
    repetitions=1000,
    error_rate=0,
    error_mitigation=True,
    layer=None,
):
    if error_rate == 0.0:
        error_mitigation = False
    n = len(x)
    circuit = create_circuit(x, y)
    input_state = pcvl.BasicState([1] + [0] * (n - 1))
    sim_processor = pcvl.Processor("SLOS", circuit)
    sim_processor.with_input(input_state)
    sampler = pcvl.algorithm.Sampler(sim_processor)
    results = sampler.sample_count(repetitions)
    if error_mitigation:
        states = [[0] * k + [1] + [0] * (n - k - 1) for k in range(n)]
        acceptable_states = [pcvl.BasicState(state) for state in states]
        num = sum(results["results"][state] for state in acceptable_states)
    else:
        num = repetitions
    total = results["results"][input_state]
    inner = np.sqrt(total / num)
    res = layer()
    if res[1] > 0.05:
        print("res", res[0], "inner", inner, inner == res[0])
    return res[0]


def quantum_distance(
    x,
    y,
    repetitions=1000,
    error_rate=0.0,
    error_mitigation=True,
):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    inner = quantum_inner(
        x,
        y,
        repetitions=repetitions,
        error_rate=error_rate,
        error_mitigation=error_mitigation,
    )
    dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
    return dist


def quantum_distance_ml(
    x,
    y,
    layer=None,
    shots=None,
    sampling_method=None,
):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    inner = quantum_inner_merlin(
        x, y, layer=layer, shots=shots, sampling_method=sampling_method
    )
    dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner.item())
    return dist


def _quantum_inner(x, y, repetitions=1000):
    qubits = cirq.LineQubit.range(len(x))
    gates_x = cirq.decompose(VectorLoader(x)(*qubits))
    gates_y = cirq.decompose(VectorLoader(y, x_flip=False)(*qubits) ** -1)
    circuit = cirq.Circuit(gates_x + gates_y, cirq.measure(qubits[0]))
    simulator = cirq.Simulator()
    measurements = simulator.run(circuit, repetitions=repetitions)
    total = measurements.histogram(key="0")[1]
    inner = np.sqrt(total / repetitions)
    return inner


def _quantum_distance(x, y, repetitions=1000):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * quantum_inner(x, y))
    return dist
