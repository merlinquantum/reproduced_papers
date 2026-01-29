import merlin as ml

# TODO: DUAL RAIL ENCODDING WITH MZI FOR THE STATE CONVERGENCE


def angle_encoding_ml(
    num_features: int,
    output_size: int = 5,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    n_photons = num_features // 2
    input_state = [0] * num_features
    for i in range(2 * num_features):
        if i % 2 == 1:
            input_state[i] = 1
    return ml.QuantumLayer(
        input_size=num_features,  # Follow the convention?
        output_size=output_size,
        input_state=input_state,
        n_photons=n_photons,
        measurement_strategy=measurement_strategy,
    )


def amplitude_encoding_ml(
    num_features: int,
    output_size: int = 5,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.AMPLITUDES,
) -> ml.QuantumLayer:
    pass
