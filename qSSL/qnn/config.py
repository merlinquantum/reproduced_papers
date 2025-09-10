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

from activation_function.activation_function import ActivationFunction
from activation_function.activation_function_factory import ActivationFunctionFactory
from ansatz.variational_ansatz import VariationalAnsatz
from ansatz.variational_ansatz_factory import VariationalAnsatzFactory
from input.data_handler import DataHandler
from input.data_handler_factory import DataHandlerFactory
from qiskit.exceptions import QiskitError
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.exceptions import IBMAccountError, IBMRuntimeError


class Config:
    def __init__(
        self,
        encoding,
        data_handler_method=None,
        ansatz_type=None,
        layers=3,
        sweeps_per_layer=1,
        activation_function_type=None,
        meas_method="all",
        grad_method="parameter shift",
        backend_type="qasm_simulator",
    ):
        """
        :param encoding:
        :param data_handler_method:
        :param ansatz_type:
        :param layers: Number of layers in variational ansatz
        :param sweeps_per_layer: Number of sweeps of parameterised gates within a single layer
        :param activation_function_type: Valid types are null, 'partial_measurement_half or 'partial_measurement_X'
        where X is an integer giving the number of measurements for the activation function.
        :param meas_method: Valid methods are 'all' or 'ancilla'
        :param grad_method:
        """
        self.encoding = encoding
        self.data_handler_method = data_handler_method
        self.ansatz_type = ansatz_type
        self.layers = layers
        self.sweeps_per_layer = sweeps_per_layer
        self.activation_function_type = activation_function_type
        self.meas_method = meas_method
        self.grad_method = grad_method
        self.backend = self.get_backend(backend_type)
        self.data_handler: DataHandler = DataHandlerFactory(
            encoding, data_handler_method
        ).get()
        self.activation_function: ActivationFunction = ActivationFunctionFactory(
            activation_function_type
        ).get()
        self.ansatz: VariationalAnsatz = VariationalAnsatzFactory(
            ansatz_type, layers, sweeps_per_layer, self.activation_function
        ).get()

    def log_info(self):
        logging.info(
            f"QNN configuration:\nencoding = {self.encoding}"
            + f"\ndata handler = {self.data_handler_method}"
            + f"\nansatz = {self.ansatz_type}"
            + f"\nnumber of layers = {self.layers}"
            + f"\nnumber of sweeps per layer = {self.sweeps_per_layer}"
            + f"\nactivation function = {self.activation_function_type}"
            + f"\noutput measurement = {self.meas_method}"
            + f"\ngradient method = {self.grad_method}"
            + f"\nsimulation backend = {self.backend}"
        )

    def get_backend(self, backend_name):
        """
        Get a backend - either from IBM Quantum or local Aer simulators.

        :param backend_name: Name of the backend to retrieve
        :return: Backend instance
        """
        backend = None

        # First check if it's a local simulator
        if backend_name in ["qasm_simulator", "statevector_simulator"]:
            try:
                backend = Aer.get_backend(backend_name)
                return backend
            except QiskitError as e:
                logging.error(f"Failed to get Aer backend {backend_name}: {e}")
                return None

        # Try to get IBM Quantum backend
        try:
            service = QiskitRuntimeService(
                channel="ibm_quantum", instance="ibm-q-oxford/oxford/default"
            )
            backend = service.backend(backend_name)
            logging.info(
                f"Successfully connected to IBM Quantum backend: {backend_name}"
            )
            return backend

        except IBMAccountError as e:
            logging.warning(f"IBM Quantum account error: {e}")
        except IBMRuntimeError as e:
            logging.warning(f"IBM Runtime error: {e}")
        except Exception as e:
            logging.debug(
                f"Failed to connect to IBM Quantum backend {backend_name}: {e}"
            )

        # Fallback to local simulators
        try:
            # Try common Aer backends as fallback
            fallback_backends = [
                "qasm_simulator",
                "statevector_simulator",
                "aer_simulator",
            ]

            for fallback in fallback_backends:
                try:
                    backend = Aer.get_backend(fallback)
                    logging.warning(
                        f"Using fallback backend: {fallback} instead of {backend_name}"
                    )
                    return backend
                except QiskitError:
                    continue

        except Exception as e:
            logging.error(f"Failed to get any fallback backend: {e}")

        if backend is None:
            raise RuntimeError(f"Could not find or access backend: {backend_name}")

        return backend
