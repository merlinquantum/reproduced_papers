import sys
from pathlib import Path

import numpy as np
import torch

try:
    from papers.DQNN.lib.classical_utils import CNNModel
    from papers.DQNN.lib.photonic_qt_utils import (
        generate_qubit_states_torch,
        probs_to_weights,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = Path(__file__).resolve().parents[3]
    for path in (REPO_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from papers.DQNN.lib.classical_utils import CNNModel
    from papers.DQNN.lib.photonic_qt_utils import (
        generate_qubit_states_torch,
        probs_to_weights,
    )


def test_generate_qubit_states_torch():
    for i in range(1, 11):
        all_states = generate_qubit_states_torch(i)
        assert len(all_states) == 2**i
        assert len(set(all_states)) == 2**i


def test_probs_to_weights():
    model_template = CNNModel(use_weight_sharing=False, shared_rows=10)
    state_dict = probs_to_weights(torch.tensor(np.random.random(6690)), model_template)
    model_params = dict(model_template.named_parameters())
    for key, value in state_dict.items():
        param_values = model_params[key]
        print(param_values)
        print(value)
        assert len(param_values) == len(value)
