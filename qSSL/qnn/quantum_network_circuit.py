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
from math import log

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

# Conditional imports for different Qiskit versions
try:
    # For Qiskit 1.0+, primitives are in qiskit.primitives
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler

    PRIMITIVES_AVAILABLE = True
    Sampler = StatevectorSampler
    Estimator = StatevectorEstimator
except ImportError:
    try:
        # For older versions, try the old location
        from qiskit.primitives import Estimator, Sampler

        PRIMITIVES_AVAILABLE = True
    except ImportError:
        # Very old versions don't have primitives
        PRIMITIVES_AVAILABLE = False
        Sampler = None
        Estimator = None
from config import Config
from qiskit.quantum_info import Pauli, Statevector

logging.basicConfig(level=logging.INFO)
logging.getLogger("qiskit").setLevel(logging.WARN)


class QuantumNetworkCircuit:
    """
    A quantum neural network. Combines state preparation circuit and variational ansatz to produce quantum neural
    network circuit.
    """

    def __init__(self, config: Config, input_qubits, input_data=None):
        self.config = config
        self.input_qubits = input_qubits
        self.input_data = input_data
        self.input_circuit_parameters = None

        self.ansatz_circuit = self._create_ansatz_circuit(input_qubits)

        self.ansatz_circuit_parameters = sorted(
            self.ansatz_circuit.parameters,
            key=lambda p: int("".join(filter(str.isdigit, p.name))),
        )

        self.qr = QuantumRegister(self.ansatz_circuit.num_qubits, name="qr")
        self.cr = ClassicalRegister(len(self.qr), name="cr")
        self.qc = QuantumCircuit(self.qr, self.cr)

        self.backend = config.backend

        if input_data is not None:
            self.construct_network(input_data)

        self.statevectors = []
        self.gradients = []
        self.transpiled = False

    def create_input_circuit(self):
        return self.config.data_handler.get_quantum_circuit(self.input_data)

    def _create_ansatz_circuit(self, input_qubits):
        return self.config.ansatz.get_quantum_circuit(input_qubits)

    def construct_network(self, input_data):
        self.input_data = input_data
        input_circuit = self.config.data_handler.get_quantum_circuit(input_data)
        self.input_circuit_parameters = sorted(
            input_circuit.parameters,
            key=lambda p: int("".join(filter(str.isdigit, p.name))),
        )

        self.qc.append(input_circuit, self.qr[: input_circuit.num_qubits])
        self.qc = self.qc.compose(self.ansatz_circuit)

        # Check backend type for measurement strategy
        backend_name = getattr(self.backend, "name", str(self.backend))
        if "statevector" not in backend_name.lower():
            if self.config.meas_method == "ancilla":
                self.qc.measure(self.qr[-1], self.cr[0])
            elif self.config.meas_method == "all":
                self.qc.measure(self.qr, self.cr)

        logging.info(
            f"QNN created with {len(self.ansatz_circuit_parameters)} trainable parameters."
        )
        self.config.log_info()

    def bind_circuit(self, parameter_values):
        """
        Assigns all parameterized gates to values
        :param parameter_values: List of parameter values for circuit. Input parameters should come before ansatz
        parameters.
        """
        if self.input_circuit_parameters is None:
            raise NotImplementedError(
                "No input data was specified before binding. Please call construct_network() first."
            )
        combined_parameter_list = (
            self.input_circuit_parameters + self.ansatz_circuit_parameters
        )
        if len(parameter_values) != len(combined_parameter_list):
            raise ValueError(
                f"Parameter_values must be of length {len(combined_parameter_list)}"
            )

        binding_dict = {}
        for i, value in enumerate(parameter_values):
            binding_dict[combined_parameter_list[i]] = value

        bound_qc = self.qc.assign_parameters(binding_dict)
        return bound_qc

    def evaluate_circuit(self, parameter_list, shots=100):
        """
        Evaluate the quantum circuit using modern Qiskit execution.
        """
        circuit = self.bind_circuit(parameter_list)

        # Check if backend supports statevector simulation
        backend_name = getattr(self.backend, "name", str(self.backend))

        if "statevector" in backend_name.lower():
            # For statevector backends, we can use the backend directly
            # since we don't need sampling, just the final statevector
            return self._legacy_execute(circuit, shots)
        else:
            # For other backends, use legacy execution
            return self._legacy_execute(circuit, shots)

    def _legacy_execute(self, circuit, shots=100):
        """
        Legacy execution method using backend.run() for Qiskit 1.0+.
        """
        try:
            # For Qiskit 1.0+, use backend.run() directly
            transpiled_circuit = transpile(circuit, self.backend)
            job = self.backend.run(transpiled_circuit, shots=shots)
            return job.result()
        except Exception as e:
            logging.error(f"Backend execution failed: {e}")
            # Try with AerSimulator as final fallback
            try:
                from qiskit_aer import AerSimulator

                simulator = AerSimulator()
                transpiled_circuit = transpile(circuit, simulator)
                job = simulator.run(transpiled_circuit, shots=shots)
                return job.result()
            except Exception as e2:
                logging.error(f"AerSimulator fallback also failed: {e2}")
                raise RuntimeError(
                    f"All execution methods failed. Original error: {e}"
                ) from e2

    @staticmethod
    def get_vector_from_results(results, circuit_id=0):
        """
        Calculates the expectation value of individual qubits for a set of observed bitstrings. Assumes counts
        corresponding to classical register used for final measurement is final element in job counts array in
        order to exclude classical registers used for activation function measurements (if present).
        :param results: Qiskit results object (modern PrimitiveResult or legacy Result).
        :param circuit_id: For results of multiple circuits, integer labelling which circuit result to use.
        :return: A vector, where the ith element is the expectation value of the ith qubit
        """

        # Handle modern PrimitiveResult from Qiskit 1.0+ primitives
        if hasattr(results, "quasi_dists"):
            try:
                # For PrimitiveResult from Sampler
                quasi_dist = results.quasi_dists[circuit_id]

                # Convert quasi-distribution to counts format
                counts = {}
                total_shots = 0
                for outcome, probability in quasi_dist.items():
                    # Convert integer outcome to binary string
                    num_qubits = len(
                        bin(max(quasi_dist.keys()))[2:]
                    )  # Get number of qubits
                    bitstring = format(outcome, f"0{num_qubits}b")
                    count = int(probability * 1000)  # Scale probability to counts
                    counts[bitstring] = count
                    total_shots += count

                if not counts:
                    raise ValueError("No measurement data found in PrimitiveResult")

                # Process counts to calculate expectation values
                first_key = next(iter(counts.keys()))
                num_measurements = len(first_key)
                vector = np.zeros(num_measurements)

                for bitstring, frequency in counts.items():
                    for i in range(num_measurements):
                        if bitstring[i] == "0":
                            vector[i] += frequency
                        elif bitstring[i] == "1":
                            vector[i] -= frequency
                        else:
                            raise ValueError(
                                f"Measurement returned unrecognised value: {bitstring[i]}"
                            )

                return vector / total_shots

            except Exception as e:
                logging.warning(
                    f"PrimitiveResult processing failed: {e}, trying legacy approach"
                )

        # Check if this is a statevector result (legacy)
        backend_name = getattr(results, "backend_name", "")

        if "statevector" in backend_name.lower():
            try:
                state = results.get_statevector(circuit_id)
                n = int(log(len(state), 2))

                # Updated Pauli construction for modern Qiskit
                vector = []
                for i in range(n):
                    # Create Pauli operator for Z measurement on qubit i
                    pauli_str = "I" * n
                    pauli_str = pauli_str[:i] + "Z" + pauli_str[i + 1 :]
                    pauli_op = Pauli(pauli_str)
                    expectation_val = (
                        Statevector(state).expectation_value(pauli_op).real
                    )
                    vector.append(expectation_val)

                return vector

            except Exception as e:
                # Fallback if statevector approach fails
                print(f"Statevector processing failed: {e}, falling back to counts")

        # Handle count-based results (legacy)
        try:
            counts = results.get_counts(circuit_id)

            # Try using get_subsystems_counts if available (legacy)
            try:
                from qiskit.aqua.utils import get_subsystems_counts

                all_register_counts = get_subsystems_counts(counts)
                output_register_counts = all_register_counts[-1]
                num_measurements = len(next(iter(output_register_counts)))
                vector = np.zeros(num_measurements)

                for count_str, frequency in output_register_counts.items():
                    for i in range(num_measurements):
                        if count_str[i] == "0":
                            vector[i] += frequency
                        elif count_str[i] == "1":
                            vector[i] -= frequency
                        else:
                            raise ValueError(
                                f"Measurement returned unrecognised value: {count_str[i]}"
                            )

                return vector / sum(output_register_counts.values())

            except ImportError:
                # Modern approach: process counts directly
                if not counts:
                    raise ValueError("No measurement counts found") from None

                # Get number of qubits from bitstring length
                first_key = next(iter(counts.keys()))
                num_measurements = len(first_key)
                vector = np.zeros(num_measurements)
                total_shots = sum(counts.values())

                for bitstring, frequency in counts.items():
                    for i in range(num_measurements):
                        if bitstring[i] == "0":
                            vector[i] += frequency
                        elif bitstring[i] == "1":
                            vector[i] -= frequency
                        else:
                            raise ValueError(
                                f"Measurement returned unrecognised value: {bitstring[i]}"
                            ) from None

                return vector / total_shots

        except Exception as e:
            raise RuntimeError(f"Failed to process measurement results: {e}") from e
