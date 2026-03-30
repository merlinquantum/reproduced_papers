from perceval import BS, PS, Circuit, P


def circuit_func(x):
    """Function for generating rectangular genericInterferometer
    with an MZI depth equal to the number of modes
    """
    circuit = Circuit(2) // PS(P(f"phi{2 * x}")) // BS()
    circuit.add(0, PS(P(f"phi{2 * x + 1}")))
    circuit.add(0, BS())
    return circuit
