import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

try:
    from papers.DQNN.lib.model import PhotonicQuantumTrain
    from papers.DQNN.lib.photonic_qt_utils import calculate_qubits
    from papers.DQNN.lib.torchmps import MPS
    from papers.DQNN.utils.utils import create_datasets
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = Path(__file__).resolve().parents[3]
    for path in (REPO_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from papers.DQNN.lib.model import PhotonicQuantumTrain
    from papers.DQNN.lib.photonic_qt_utils import calculate_qubits
    from papers.DQNN.lib.torchmps import MPS
    from papers.DQNN.utils.utils import create_datasets


@pytest.fixture
def create_model():
    return PhotonicQuantumTrain(calculate_qubits()[0])


def test_calculate_qubits():
    n_qubits, nw_list = calculate_qubits()
    assert n_qubits == 13
    assert len(nw_list) == 6690


def test_init(create_model):
    model = create_model
    assert isinstance(model, nn.Module)
    assert isinstance(model.MappingNetwork, MPS)
    assert model.MappingNetwork.bond_dim == 7
    assert model.MappingNetwork.input_dim == calculate_qubits()[0] + 1
    assert model.MappingNetwork.output_dim == 1


def test_foward(create_model, bs_1, bs_2):
    model = create_model
    model.train()
    n_qubits, nw_list_norm = calculate_qubits()
    train_dataset = create_datasets()[2]
    index = 0
    for i, (images, _) in enumerate(train_dataset):
        outputs = model(
            images,
            bs_1=bs_1,
            bs_2=bs_2,
            n_qubit=n_qubits,
            nw_list_normal=nw_list_norm,
        )
        _, predicted = torch.max(outputs.data, 1)
        if i < len(train_dataset) - 1:
            assert len(predicted) == 128
        index += 1
        for label in predicted:
            assert label < 10
            assert label >= 0
