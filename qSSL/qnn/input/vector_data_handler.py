from input.data_handler import DataHandler
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class VectorDataHandler(DataHandler):
    def get_quantum_circuit(self, input_data):
        self.qr = QuantumRegister(len(input_data))
        self.qc = QuantumCircuit(self.qr)

        for index, _ in enumerate(input_data):
            param = Parameter(f"input{str(index)}")
            self.qc.rx(param, index)

        return self.qc
