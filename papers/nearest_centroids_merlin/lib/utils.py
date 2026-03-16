import numpy as np
import perceval as pcvl


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def get_angles(x):
    # convert to array
    x = np.array(x)
    shape = x.shape
    if len(shape) == 1:
        x = np.expand_dims(x, axis=0)

    if x.shape[1] == 1:
        x = x.T

    # get recursively the angles
    def angles(y, wire):
        d = y.shape[-1]
        if d == 2:
            thetas = np.arccos(y[:, 0] / np.linalg.norm(y, 2, 1))
            signs = (y[:, 1] > 0.0).astype(int)
            thetas = signs * thetas + (1.0 - signs) * (2.0 * np.pi - thetas)
            thetas = np.expand_dims(thetas, 1)
            wires = [(wire, wire + 1)]
            return thetas, wires
        thetas = np.arccos(
            np.linalg.norm(y[:, : d // 2], 2, 1, True) / np.linalg.norm(y, 2, 1, True)
        )
        thetas_l, wires_l = angles(y[:, : d // 2], wire)
        thetas_r, wires_r = angles(y[:, d // 2 :], wire + d // 2)
        thetas = np.concatenate([thetas, thetas_l, thetas_r], axis=1)
        wires = [(wire, wire + d // 2)] + wires_l + wires_r
        return thetas, wires

    # result
    thetas, wires = angles(x, 0)

    # remove nan and one dims
    thetas = np.nan_to_num(thetas, nan=0)
    thetas = thetas.squeeze()

    return thetas, wires


def _decompose_(self, qubits):
    # NOTE: leftover helper; not used directly in this reproduction.
    if self.x_flip:
        import cirq  # local import to avoid hard dependency at module level

        yield cirq.X(qubits[0])
    for theta, (i, j) in zip(self.thetas, self.wires):
        from .gates import RBS  # local import

        yield RBS(theta)(qubits[i], qubits[j])


def create_circuit(n):
    x = np.random.rand(n)
    y = np.random.rand(n)
    angle_x, wires_x = get_angles(x)
    angle_y, wires_y = np.flip(get_angles(y)[0], 0), np.flip(get_angles(y)[1], 0)
    circuit = pcvl.Circuit(n)
    k = 0
    for _theta, (i, j) in zip(angle_x, wires_x):
        px = pcvl.P(f"theta_{k}")
        k += 1
        circuit.add(i, pcvl.BS(theta=px))
        perm = [j - i - 1] + list(range(j - i - 1))
        circuit.add(i + 1, pcvl.PERM(perm))

    for _theta, (i, j) in zip(angle_y, wires_y):
        i = int(i)
        j = int(j)
        perm = list(range(1, j - i)) + [0]
        circuit.add(i + 1, pcvl.PERM(perm))
        px = pcvl.P(f"theta_{k}")
        k += 1
        circuit.add(i, pcvl.BS(theta=px))

    return circuit
