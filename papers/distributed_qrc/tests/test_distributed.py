"""An MRMR model computed branch by branch equals the runner's sequential computation."""

from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from common import load_runtime_ready_config
from distributed_qrc.lib import runner
from distributed_qrc.lib.architectures import build_cell_spec
from distributed_qrc.lib.data import RAW_SUBDIR, build_forecast_data, dataset_dir
from distributed_qrc.lib.distributed import branch_forecast
from distributed_qrc.lib.noise import devices_for_components
from distributed_qrc.lib.readouts import merge_predictions

CFG = load_runtime_ready_config()
pytestmark = pytest.mark.skipif(
    not (dataset_dir(CFG) / RAW_SUBDIR).is_dir()
    and not (dataset_dir(CFG) / "processed").is_dir(),
    reason="IPTO data not acquired; see the README Data section",
)


@pytest.mark.parametrize("readout_type", ["linear", "quantum_kernel_ideal"])
def test_branches_reproduce_the_sequential_mrmr_forecast(readout_type):
    cfg = dict(CFG, seed=0, device="cpu")
    data = build_forecast_data(cfg)
    spec = build_cell_spec("mrmr", "2x5")
    mapping = cfg["noise"]["mapping"]
    features = runner._reservoir_features(
        cfg, data, spec, 0, "quantum_ideal", devices_for_components(2, mapping)
    )
    sequential, _ = runner._fit_cell(
        cfg,
        data,
        spec,
        features,
        readout_type,
        devices_for_components(spec.n_instances, mapping),
        f"mrmr:2x5|quantum_ideal|{readout_type}",
    )
    branches = [
        branch_forecast(cfg, data, spec, 0, b, readout_type=readout_type)["test"]
        for b in range(2)
    ]
    distributed = merge_predictions(branches, cfg["readout"]["merge"])
    np.testing.assert_allclose(distributed, sequential["test"], rtol=0, atol=1e-10)
