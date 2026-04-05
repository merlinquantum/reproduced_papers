from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from HQPINN.lib.DEE import core_dee
from HQPINN.lib.DHO import dho_cc
from HQPINN.lib.SEE import core_see
from HQPINN.lib.TAF import core_taf


class _DummyTAFModel(torch.nn.Module):
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        rho = torch.full_like(x, 1.225)
        u = 300.0 + 5.0 * x
        v = 10.0 * y
        temp = torch.full_like(x, 288.15)
        return torch.cat([rho, u, v, temp], dim=1)


class _DummyDEEModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, dtype=core_dee.DTYPE)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raw = self.linear(xt)
        rho = raw[:, 0:1] * 0.0 + 1.2
        vel = raw[:, 1:2] * 0.0 + 0.1
        pressure = raw[:, 2:3] * 0.0 + 1.0
        return torch.cat([rho, vel, pressure], dim=1)


class _DummySEEModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, dtype=core_see.DTYPE)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raw = self.linear(xt)
        rho = raw[:, 0:1] * 0.0 + 1.0
        vel = raw[:, 1:2] * 0.0 + 1.0
        pressure = raw[:, 2:3] * 0.0 + 1.0
        return torch.cat([rho, vel, pressure], dim=1)


class PlotOutputTests(unittest.TestCase):
    def test_train_taf_saves_figure7_outputs_into_case_dir(self) -> None:
        model = torch.nn.Linear(2, 4, bias=False, dtype=core_taf.DTYPE)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        fake_plot_paths = ["figure7.png", "rho.png"]

        def _fake_boundary_terms(*args, **kwargs):
            base = model.weight.sum() * 0.0
            return base + 1.0, base + 2.0, base + 3.0, base + 4.0

        def _fake_loss_pde(*args, **kwargs):
            return model.weight.sum() * 0.0 + 5.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(core_taf, "loss_boundary_terms", side_effect=_fake_boundary_terms),
                patch.object(core_taf, "loss_pde", side_effect=_fake_loss_pde),
                patch.object(
                    core_taf,
                    "_save_primitive_field_plots",
                    return_value=fake_plot_paths,
                ) as mocked_save_plots,
                patch.object(core_taf, "count_trainable_params", return_value=8),
            ):
                core_taf.train_taf(
                    model=model,
                    optimizer=optimizer,
                    n_epochs=1,
                    plot_every=1,
                    out_dir=tmp_dir,
                    model_label="cc_40-4",
                    run_id="run123",
                    data={},
                    U_in=torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=core_taf.DTYPE),
                    lbfgs_steps=0,
                )

            mocked_save_plots.assert_called_once_with(
                model=model,
                data={},
                output_dir=tmp_dir,
                filename_prefix="taf-cc_40-4_run123",
            )
            self.assertTrue((Path(tmp_dir) / "taf-cc_40-4_run123.csv").is_file())

    def test_taf_save_density_plot_writes_into_case_specific_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            main_png = Path(tmp_dir) / "taf_cc_40-4_local_run123_figure7_pred.png"
            extra_png = Path(tmp_dir) / "taf_cc_40-4_local_run123_rho_pred.png"

            def _fake_save_primitive_field_plots(**kwargs):
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                main_png.write_bytes(b"png")
                extra_png.write_bytes(b"png")
                return [str(main_png), str(extra_png)]

            with (
                patch.object(core_taf, "load_training_sets", return_value={}),
                patch.object(
                    core_taf,
                    "results_case_dir_for_model_dir",
                    return_value=tmp_dir,
                ),
                patch.object(
                    core_taf,
                    "_save_primitive_field_plots",
                    side_effect=_fake_save_primitive_field_plots,
                ),
            ):
                png_path = core_taf.save_density_plot(
                    model=_DummyTAFModel(),
                    ckpt_dir="HQPINN/models/TAF",
                    case_prefix="taf_cc_40-4",
                    plot_label=None,
                    run_id="run123",
                    backend="local",
                )

            self.assertEqual(Path(png_path), main_png)
            self.assertTrue(main_png.is_file())
            self.assertTrue(extra_png.is_file())

    def test_train_dee_saves_rho_slice_into_case_dir(self) -> None:
        model = _DummyDEEModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def _fake_loss_fn(*args, **kwargs):
            base = model.linear.weight.sum() * 0.0
            return base + 1.0, base + 2.0, base + 3.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(core_dee, "DEE_NX_SAMPLES", 8),
                patch.object(core_dee, "DEE_NT_SAMPLES", 6),
                patch.object(core_dee, "evaluate_dee_errors", return_value=(0.0, 0.0)),
            ):
                core_dee.train_dee(
                    model=model,
                    t_train=None,
                    optimizer=optimizer,
                    n_epochs=1,
                    plot_every=1,
                    out_dir=tmp_dir,
                    model_label="cc_10-4",
                    run_id="run123",
                    loss_fn=_fake_loss_fn,
                )

            self.assertTrue((Path(tmp_dir) / "dee-cc_10-4_run123_rho_pred.png").is_file())
            self.assertTrue((Path(tmp_dir) / "dee-cc_10-4_run123_rho_exact.png").is_file())
            self.assertTrue((Path(tmp_dir) / "dee-cc_10-4_run123_rho_error.png").is_file())
            self.assertTrue((Path(tmp_dir) / "dee-cc_10-4_run123_rho_x_t_2p0.png").is_file())

    def test_dee_save_density_plot_writes_main_and_slice_into_case_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(core_dee, "DEE_NX_SAMPLES", 8),
                patch.object(core_dee, "DEE_NT_SAMPLES", 6),
                patch.object(
                    core_dee,
                    "results_case_dir_for_model_dir",
                    return_value=tmp_dir,
                ),
            ):
                png_path = core_dee.save_density_plot(
                    model=_DummyDEEModel(),
                    ckpt_dir="HQPINN/models/DEE",
                    case_prefix="dee_cc_10-4",
                    plot_label=None,
                    run_id="run123",
                    backend="local",
                )

            self.assertTrue(Path(png_path).is_file())
            self.assertTrue(
                (Path(tmp_dir) / "dee_cc_10-4_local_run123_rho_x_t_2p0.png").is_file()
            )

    def test_see_save_density_plot_writes_into_case_specific_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(core_see, "SEE_NX_SAMPLES", 8),
                patch.object(core_see, "SEE_NT_SAMPLES", 6),
                patch.object(
                    core_see,
                    "results_case_dir_for_model_dir",
                    return_value=tmp_dir,
                ),
            ):
                png_path = core_see.save_density_plot(
                    model=_DummySEEModel(),
                    ckpt_dir="HQPINN/models/SEE",
                    case_prefix="see_cc_10-4",
                    plot_label=None,
                    run_id="run123",
                    backend="local",
                )

            self.assertTrue(Path(png_path).is_file())
            self.assertEqual(
                Path(png_path),
                Path(tmp_dir) / "see_cc_10-4_local_run123.png",
            )

    def test_dho_cc_run_mode_uses_case_specific_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            captured = {}

            def _fake_run_series_inference_mode(**kwargs):
                captured["case_prefix"] = kwargs["case_prefix"]
                t = torch.linspace(0.0, 1.0, 4, dtype=dho_cc.DTYPE).view(-1, 1)
                u_pred = torch.zeros(4)
                u_ex = torch.zeros(4)
                kwargs["plot_fn"](u_pred, u_ex, t)

            with (
                patch.object(
                    dho_cc,
                    "results_case_dir_for_model_dir",
                    return_value=tmp_dir,
                ),
                patch.object(
                    dho_cc,
                    "run_series_inference_mode",
                    side_effect=_fake_run_series_inference_mode,
                ),
            ):
                dho_cc.run(mode="run", backend="local")

            self.assertEqual(captured["case_prefix"], "dho_cc")
            self.assertEqual(len(list(Path(tmp_dir).glob("dho_cc_plot_*.png"))), 1)
