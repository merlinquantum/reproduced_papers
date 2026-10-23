from __future__ import annotations

import pytest
from lib.models import (
    LSTMClassifier,
    MLPClassifier,
    QLSTMClassifier,
    QNNClassifier,
    count_parameters,
)


@pytest.mark.parametrize(
    "n_qubits, layers, expected", [(4, 2, 233), (5, 4, 476), (5, 3, 386), (5, 6, 656)]
)
def test_qlstm_matches_table_xii_parameter_counts(n_qubits, layers, expected):
    assert count_parameters(QLSTMClassifier(4, n_qubits, layers)) == expected


@pytest.mark.parametrize(
    "hidden, layers, expected", [(4, 2, 325), (5, 6, 1426), (4, 4, 645)]
)
def test_lstm_matches_table_xii_parameter_counts(hidden, layers, expected):
    assert count_parameters(LSTMClassifier(4, hidden, layers)) == expected


@pytest.mark.parametrize(
    "features, hidden, expected", [(3, [11], 56), (7, [32, 16], 801), (64, [32], 2113)]
)
def test_ann_baselines_match_section_iii_c3(features, hidden, expected):
    assert count_parameters(MLPClassifier(features, hidden)) == expected


def test_non_hybrid_qnn_rejects_wrong_feature_count():
    with pytest.raises(ValueError):
        QNNClassifier(
            5,
            3,
            1,
            hybrid=False,
            multi_qubit_readout=False,
            encoding="angle",
            angle_scale=1.0,
        )


def test_qnn_parameter_count_formula():
    model = QNNClassifier(
        64,
        5,
        6,
        hybrid=True,
        multi_qubit_readout=False,
        encoding="amplitude",
        angle_scale=1.0,
    )
    assert count_parameters(model) == 64 * 32 + 32 + 3 * 5 * 6
