from __future__ import annotations

import pytest
import torch
from lib.autoencoder import Decoder, Encoder
from lib.gqc_model import GQCModel
from lib.vqc_classical import matched_hidden_dim
from lib.vqc_photonic import spread_input_state


@pytest.mark.parametrize("backend", ["gate", "photonic", "classical"])
def test_gqc_model_forward_shapes_across_backends(backend):
    input_dim = 29
    batch = 4
    cfg = {"model": {"backend": backend, "n_qubits": 6}}
    model = GQCModel(input_dim=input_dim, cfg=cfg)
    x = torch.rand(batch, input_dim)

    p_hat, x_recon = model(x)
    assert p_hat.shape == (batch,)
    assert x_recon.shape == (batch, input_dim)
    assert torch.all((p_hat >= 0) & (p_hat <= 1))

    p_infer = model.predict_proba(x)
    assert p_infer.shape == (batch,)


def test_photonic_layer_rejects_single_photon():
    with pytest.raises(ValueError):
        spread_input_state(n_modes=6, n_photons=1)


def test_photonic_input_state_spreads_photons_and_conserves_count():
    state = spread_input_state(n_modes=6, n_photons=3)
    assert sum(state) == 3
    assert len(state) == 6


@pytest.mark.parametrize("readout", ["fixed", "trainable"])
def test_photonic_readout_modes(readout):
    input_dim = 29
    batch = 4
    cfg = {
        "model": {
            "backend": "photonic",
            "n_qubits": 6,
            "photonic": {"n_photons": 3, "readout": readout},
        }
    }
    model = GQCModel(input_dim=input_dim, cfg=cfg)
    x = torch.rand(batch, input_dim)
    p_hat, _ = model(x)
    assert p_hat.shape == (batch,)
    assert torch.all((p_hat >= 0) & (p_hat <= 1))
    loss = p_hat.sum()
    loss.backward()
    if readout == "trainable":
        assert model.vqc.grouping[0].weight.grad is not None


def test_matched_hidden_dim_meets_or_exceeds_target():
    n_qubits = 6
    target = 72  # gate VQC: n_layers=6 * n_qubits=6 * 2 angles
    hidden = matched_hidden_dim(n_qubits, target)
    actual_params = hidden * (n_qubits + 2) + 1
    assert actual_params >= target
    # Should be the smallest such hidden dim (not wastefully larger).
    assert (hidden - 1) * (n_qubits + 2) + 1 < target


def test_encoder_rejects_latent_not_smaller_than_input():
    """The latent must compress: the paper specifies a lower latent dimension."""
    with pytest.raises(ValueError, match="strictly smaller"):
        Encoder(input_dim=6, hidden_dims=[32], latent_dim=6)
    with pytest.raises(ValueError, match="strictly smaller"):
        Encoder(input_dim=6, hidden_dims=[32], latent_dim=8)


def test_decoder_rejects_latent_not_smaller_than_output():
    with pytest.raises(ValueError, match="strictly smaller"):
        Decoder(output_dim=6, hidden_dims=[32], latent_dim=6)


@pytest.mark.parametrize("bad_latent", [0, -1])
def test_autoencoder_rejects_nonpositive_latent(bad_latent):
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        Encoder(input_dim=29, hidden_dims=[32], latent_dim=bad_latent)


def test_autoencoder_accepts_the_papers_architecture():
    """29 -> 256 -> 128 -> 64 -> 6 must still build.

    The hidden layers are wider than the 29-feature input, so the check must
    constrain the latent against the input only, not require a funnel.
    """
    encoder = Encoder(input_dim=29, hidden_dims=[256, 128, 64], latent_dim=6)
    decoder = Decoder(output_dim=29, hidden_dims=[256, 128, 64], latent_dim=6)
    z = encoder(torch.rand(4, 29))
    assert z.shape == (4, 6)
    assert decoder(z).shape == (4, 29)
