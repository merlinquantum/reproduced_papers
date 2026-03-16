from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from sklearn.neighbors import NearestCentroid

from .functions import quantum_distance, quantum_distance_ml
from .utils import create_circuit


class MLQuantumNearestCentroid(NearestCentroid):
    def __init__(self, repetitions=500, error_rate=0.0, error_mitigation=True, n=8):
        self.repetitions = repetitions
        self.error_rate = error_rate
        self.error_mitigation = error_mitigation
        self.n = n

        input_state = [1] + [0] * (n - 1)
        circuit = create_circuit(n)
        self.layer = QuantumLayer(
            input_size=2 * (n - 1),
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["theta"],
            input_state=input_state,
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        super().__init__(metric=self.get_metric)

    def get_metric(self, x, y):
        return quantum_distance_ml(
            x,
            y,
            layer=self.layer,
            shots=self.repetitions,
            sampling_method="multinomial",
        )

    def fit(self, X, y):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=(
                    "Averaging for metrics other than "
                    "euclidean and manhattan not supported. "
                    "The average is set to be the mean."
                ),
            )
            return super().fit(X, y)


class QuantumNearestCentroid(NearestCentroid):
    def __init__(self, repetitions=1000, error_rate=0.0, error_mitigation=True):
        self.repetitions = repetitions
        self.error_rate = error_rate
        self.error_mitigation = error_mitigation
        super().__init__(metric=self.get_metric)

    def get_metric(self, x, y):
        return quantum_distance(
            x,
            y,
            repetitions=self.repetitions,
            error_rate=self.error_rate,
            error_mitigation=self.error_mitigation,
        )

    def fit(self, X, y):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=(
                    "Averaging for metrics other than "
                    "euclidean and manhattan not supported. "
                    "The average is set to be the mean."
                ),
            )
            return super().fit(X, y)
