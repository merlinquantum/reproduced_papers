"""Pin the QRC simulator against the authors' published artefacts.

The authors' repository ships ``predict_result.csv``, the 245 out-of-sample QR1
and QR2 forecasts that produced paper Table II. Reproducing those forecasts to
``float32`` precision (the reference Julia code uses ``ComplexF32``) is the
strongest available check that the Python reimplementation of the reservoir,
encoding, partial trace, virtual nodes and ridge readout is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.data import (
    build_lagged_inputs,
    build_regressor_frame,
    denormalise_log_rv,
    load_coupling_instances,
    load_normalised_table,
    rolling_windows,
)
from lib.metrics import mse, qlike
from lib.qrc import reservoir_readout, rolling_ridge_forecast

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "qrc_volatility"
N_LAGS = 3
N_OUT_OF_SAMPLE = 245

VARIANTS = {
    # variant: (features, virtual_nodes, coupling instance, paper MSE, paper QLIKE)
    "QR1": (["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"], 1, 0, 0.105, 1.4427),
    "QR2": (["RV", "MKT", "STR", "RV_q", "EP", "INF", "DEF"], 2, 1, 0.103, 1.4004),
}

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "Data.CSV").exists(),
    reason="authors' Data.CSV / coeff_10.jld2 not present in data/qrc_volatility",
)


@pytest.fixture(scope="module")
def sample():
    normalised = load_normalised_table(DATA_DIR)
    frame = build_regressor_frame(normalised)
    train_slices, origin_index = rolling_windows(len(normalised), N_OUT_OF_SAMPLE)
    return {
        "normalised": normalised,
        "target": normalised["RV"].to_numpy(),
        "actual": frame["RV"].to_numpy()[np.asarray(origin_index)],
        "couplings": load_coupling_instances(DATA_DIR),
        "train_slices": train_slices,
        "origin_index": origin_index,
    }


@pytest.fixture(scope="module")
def forecasts(sample):
    """Cache the two QRC forecast paths; each costs a full 816-month simulation."""
    out = {}
    for variant, (features, virtual_nodes, instance, _, _) in VARIANTS.items():
        windows = build_lagged_inputs(sample["normalised"], features, N_LAGS)
        readout = reservoir_readout(
            windows, sample["couplings"][instance], n_qubits=10, tau=1.0,
            virtual_nodes=virtual_nodes,
        )
        out[variant] = denormalise_log_rv(rolling_ridge_forecast(
            readout, sample["target"], sample["train_slices"], sample["origin_index"],
            delta=1e-8,
        ))
    return out


def test_coupling_instances_match_reference_construction(sample):
    couplings = sample["couplings"]
    assert couplings.shape == (100, 10, 10)
    for matrix in couplings[:5]:
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)
        # coeff_matrix normalises by the largest eigenvalue with J = 1.
        assert np.linalg.eigvalsh(matrix).max() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_forecasts_match_authors_saved_predictions(sample, forecasts, variant):
    reference_path = DATA_DIR / "authors_qr_predictions.csv"
    if not reference_path.exists():
        pytest.skip("authors_qr_predictions.csv not present")
    import pandas as pd

    reference = pd.read_csv(reference_path)[variant].to_numpy()
    predicted = forecasts[variant]
    assert predicted.shape == reference.shape
    # The reference simulation is float32; agreement at 1e-3 is far tighter than
    # any scientifically relevant tolerance here.
    np.testing.assert_allclose(predicted, reference, atol=1e-3)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_metrics_match_paper_table2(sample, forecasts, variant):
    _, _, _, paper_mse, paper_qlike = VARIANTS[variant]
    predicted = forecasts[variant]
    # Table II prints MSE to three decimals and truncates rather than rounds
    # (QR2's own saved forecasts give 0.10375, printed as "0.103"), so the MSE
    # tolerance has to admit one unit in the last printed place.
    assert mse(predicted, sample["actual"]) == pytest.approx(paper_mse, abs=1.1e-3)
    assert qlike(predicted, sample["actual"]) == pytest.approx(paper_qlike, abs=5e-4)


def test_har_reproduces_paper_value_only_with_the_reference_misalignment(sample):
    """The published HAR loss depends on an off-by-one in the authors' notebook.

    Their rolling loop reads the regressor row one month before the forecast
    target, so the published HAR MSE (0.1476) is not the loss of a correctly
    indexed HAR model. Correcting the alignment lowers it substantially.
    """
    from lib.baselines import har_forecasts

    normalised = sample["normalised"]
    frame = build_regressor_frame(normalised)
    slices, origins = sample["train_slices"], sample["origin_index"]
    actual = sample["actual"]

    as_published = har_forecasts(frame, slices, origins, (1,), reference_misalignment=True)[1]
    corrected = har_forecasts(frame, slices, origins, (1,))[1]

    assert mse(as_published, actual) == pytest.approx(0.1476, abs=5e-4)
    assert mse(corrected, actual) < mse(as_published, actual) - 0.02
