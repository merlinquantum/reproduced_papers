"""Random kitchen sink samplers used by the RF-RQKS ablation."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn


class DummyRBFSampler(nn.Module):
    """Classical random Fourier sampler.

    Parameters
    ----------
    photon_count : int
        Photon count used to determine features per episode.
    mode_count : int
        Mode count used to determine features per episode.
    depth : int
        Circuit depth retained for protocol compatibility.
    episode_count : int
        Number of independent random-feature episodes.
    input_feature_count : int
        Flattened input dimension.
    """

    def __init__(
        self,
        photon_count: int,
        mode_count: int,
        depth: int,
        episode_count: int,
        input_feature_count: int,
    ) -> None:
        super().__init__()
        self.photon_count = photon_count
        self.mode_count = mode_count
        self.depth = depth
        self.episode_count = episode_count
        self.features_per_episode = math.comb(mode_count, photon_count)
        self.output_feature_count = episode_count * self.features_per_episode
        weights = torch.randn(
            episode_count, input_feature_count, self.features_per_episode
        ) * math.sqrt(2.0)
        biases = torch.rand(episode_count, self.features_per_episode) * 2.0 * math.pi
        self.register_buffer("random_weights", weights)
        self.register_buffer("random_biases", biases)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform an input batch into random Fourier features.

        Parameters
        ----------
        features : torch.Tensor
            Input matrix with shape ``(batch, input_feature_count)``.

        Returns
        -------
        torch.Tensor
            Sampled matrix with ``E * comb(m, n)`` columns.
        """
        projected = torch.einsum("bi,eic->bec", features, self.random_weights)
        sampled = torch.cos(projected + self.random_biases)
        sampled *= math.sqrt(2.0 / self.output_feature_count)
        return sampled.reshape(features.shape[0], self.output_feature_count)


def build_sampler(
    sampler_name: str,
    photon_count: int,
    mode_count: int,
    depth: int,
    episode_count: int,
    input_feature_count: int,
    encoding_strategy: str,
    entangling_strategy: str | None,
    same_haar: bool,
    qubit_count: int | None = None,
    run_on_hardware: bool = False,
    hardware: str = "sim:slos",
    nsample: int = 5000,
    forward_saves_directory: str | None = None,
) -> nn.Module:
    """Construct the RQKS sampler.

    Parameters
    ----------
    sampler_name : str
        ``dummy_rbf``, ``qiskit``, or ``photonic``.
    photon_count : int | None
        Number of photons for the photonic backend.
    mode_count : int | None
        Number of optical modes for the photonic backend.
    qubit_count : int | None
        Number of qubits for the Qiskit backend.
    depth : int
        Circuit depth.
    episode_count : int
        Number of random episodes.
    input_feature_count : int
        Flattened DCT feature count.
    encoding_strategy : str
        Photonic encoding layer name.
    entangling_strategy : str | None
        Photonic entangling layer name. If omitted, no entangler is used.
    same_haar : bool
        Whether V1 layers reuse a Haar unitary.
    run_on_hardware : bool
        Whether the photonic sampler should submit circuits to a remote
        Perceval processor. Default value is False.
    hardware : str
        Perceval remote backend name. Default value is ``"sim:slos"``.
    nsample : int
        Number of samples requested for each remote circuit. Default value is
        5000.
    forward_saves_directory : str | None
        Directory for cached remote forward results. Default value is None.

    Returns
    -------
    torch.nn.Module
        Configured sampler.

    Raises
    ------
    ValueError
        If ``sampler_name`` is unsupported.
    """
    if sampler_name == "dummy_rbf":
        return DummyRBFSampler(
            photon_count,
            mode_count,
            depth,
            episode_count,
            input_feature_count,
        )
    if sampler_name == "photonic":
        try:
            merlin_version = version("merlinquantum")
        except PackageNotFoundError as exc:
            raise ImportError(
                "The photonic sampler requires merlinquantum==0.4.1"
            ) from exc
        if merlin_version != "0.4.1":
            raise RuntimeError(
                "The RF-RQKS photonic sampler requires merlinquantum==0.4.1; "
                f"found {merlin_version}"
            )
        from .photonic_qrks import PhotonicQRKS

        return PhotonicQRKS(
            n=photon_count,
            m=mode_count,
            D=depth,
            E=episode_count,
            L_strategy=encoding_strategy,
            V_strategy=entangling_strategy,
            data_size=input_feature_count,
            v1_same_haar_dist=same_haar,
            run_on_hardware=run_on_hardware,
            hardware=hardware,
            nsample=nsample,
            forward_saves_directory=forward_saves_directory,
        )
    if sampler_name == "qiskit":
        from .qiskit_qrks import QiskitQRKS

        if run_on_hardware:
            raise ValueError(
                "The Qiskit sampler is simulator-only and does not support run_on_hardware"
            )
        if qubit_count is None:
            raise ValueError("qubit_count is required for the Qiskit sampler")
        return QiskitQRKS(
            qubit_count=qubit_count,
            depth=depth,
            episode_count=episode_count,
            L_strategy=encoding_strategy,
            V_strategy=entangling_strategy,
            data_size=input_feature_count,
            same_random_layer=same_haar,
        )
    raise ValueError(f"Unsupported sampler: {sampler_name}")
