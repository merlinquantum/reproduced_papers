"""Smoke tests for the convolutional autoencoder."""

from __future__ import annotations

import torch
from lib.autoencoder import Autoencoder, Decoder, Encoder


def test_encoder_shape():
    enc = Encoder(T=5, NG=3)
    x = torch.rand(2, 1, 28, 28)
    z = enc(x)
    assert z.shape == (2, 5, 8)
    row_sums = z.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_decoder_shape():
    dec = Decoder(T=5, NG=3)
    z = torch.rand(2, 5, 8)
    z = z / z.sum(dim=-1, keepdim=True)
    out = dec(z)
    assert out.shape == (2, 1, 28, 28)
    assert (out >= 0).all() and (out <= 1).all()


def test_autoencoder_roundtrip():
    ae = Autoencoder(T=5, NG=3)
    x = torch.rand(2, 1, 28, 28)
    out = ae(x)
    assert out.shape == x.shape
