"""
Photonic primitives for QVT.

Every trainable interferometer is a single ``add_entangling_layer(model="mzi")``
— a rectangular GenericInterferometer, universal for U(m).

Amplitude encoding throughout: StateVector.from_tensor → QuantumLayer.forward.
"""

from __future__ import annotations

import math

import merlin as ML
import torch
import torch.nn as nn
from merlin.builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.state_vector import StateVector
from merlin.measurement import MeasurementStrategy

# ── helpers ─────────────────────────────────────────────────────────────


def normalize_for_encoding(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Map a real/complex torch dtype to the matching complex dtype."""
    if dtype in (torch.float16, torch.float32, torch.bfloat16, torch.complex64):
        return torch.complex64
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    raise TypeError(f"Unsupported dtype for amplitude encoding: {dtype}.")


def _fock_basis_size(n_modes: int, n_photons: int) -> int:
    return math.comb(n_modes + n_photons - 1, n_photons)


# ── Trainable Interferometer ────────────────────────────────────────────


class TrainableInterferometer(nn.Module):
    """m-mode interferometer via MerLin QuantumLayer + amplitude encoding."""

    def __init__(
        self,
        m: int,
        n_photons: int = 1,
        name: str = "U",
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.m = m
        self.n_photons = n_photons
        self.basis_size = _fock_basis_size(m, n_photons)

        builder = CircuitBuilder(n_modes=m)
        builder.add_entangling_layer(trainable=True, model="mzi", name=name)

        self.layer = ML.QuantumLayer(
            builder=builder,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
            return_object=True,
            device=device,
        )
        self._output_keys = list(self.layer.output_keys)

    @property
    def output_keys(self):
        return self._output_keys

    def forward_sv(self, sv: StateVector):
        return self.layer(sv)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, ..., basis_size] real → [B, ..., basis_size] real (probabilities).

        Returns |α_j|² — the photon-number probability distribution.
        This is what you physically measure on a photonic chip via
        photon-number-resolving detectors.  Non-negative, sums to 1.

        The classical MLP layers downstream have negative weights and
        can map these non-negative inputs into signed features.
        """
        shape, dtype = x.shape, x.dtype
        complex_dtype = complex_dtype_for(dtype)
        sv = StateVector.from_tensor(
            x.reshape(-1, self.basis_size).to(complex_dtype),
            n_modes=self.m,
            n_photons=self.n_photons,
        )
        result = self.layer(sv)
        amps = result.to_dense() if isinstance(result, StateVector) else result
        probs = amps.real.pow(2) + amps.imag.pow(2)
        return probs.to(dtype).reshape(shape)

    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        """[B, ..., basis_size] real → [B, ..., basis_size] complex amplitudes."""
        shape = x.shape
        complex_dtype = complex_dtype_for(x.dtype)
        sv = StateVector.from_tensor(
            x.reshape(-1, self.basis_size).to(complex_dtype),
            n_modes=self.m,
            n_photons=self.n_photons,
        )
        result = self.layer(sv)
        out = result.to_dense() if isinstance(result, StateVector) else result
        return out.reshape(*shape[:-1], self.basis_size)


# ── Overlap Estimator (Model B) ────────────────────────────────────────


class OverlapEstimator(nn.Module):
    """|⟨x_i | W | x_j⟩|² via complex amplitudes."""

    def __init__(self, W: TrainableInterferometer):
        super().__init__()
        self.W = W

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        Wx_j = self.W.forward_complex(x_j)
        overlap = torch.einsum("...id,...jd->...ij", x_i.to(Wx_j.dtype), Wx_j)
        return overlap.abs().pow(2).to(x_i.dtype)


# ── CompoundSectorReadout (2-photon, cross-partition only) ──────────────


class CompoundSectorReadout(nn.Module):
    """1 photon patch + 1 photon feature → [n, d]."""

    def __init__(self, n_patches: int, d_feat: int, output_keys: list[tuple[int, ...]]):
        super().__init__()
        self.n, self.d = n_patches, d_feat
        m = n_patches + d_feat

        vi, pi, fi = [], [], []
        for idx, key in enumerate(output_keys):
            occ = list(key)
            if sum(occ[:n_patches]) == 1 and sum(occ[n_patches:]) == 1:
                vi.append(idx)
                pi.append(next(i for i in range(n_patches) if occ[i] == 1))
                fi.append(
                    next(i for i in range(n_patches, m) if occ[i] == 1) - n_patches
                )

        self.register_buffer("vi", torch.tensor(vi, dtype=torch.long))
        self.register_buffer("pi", torch.tensor(pi, dtype=torch.long))
        self.register_buffer("fi", torch.tensor(fi, dtype=torch.long))

    def forward(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = probs.shape[:-1]
        valid = probs[..., self.vi]
        sector_mass = valid.sum(dim=-1) / (probs.sum(dim=-1) + 1e-12)
        Y = torch.zeros(*bs, self.n, self.d, dtype=probs.dtype, device=probs.device)
        Y[..., self.pi, self.fi] = valid
        Y = Y / Y.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        return Y, sector_mass


# ── FullSectorReadout (2-photon, all three sectors) ─────────────────────


class FullSectorReadout(nn.Module):
    """
    Cross-partition → [n, d], patch-patch → [n, n], feat-feat → [d, d].
    """

    def __init__(self, n_patches: int, d_feat: int, output_keys: list[tuple[int, ...]]):
        super().__init__()
        self.n, self.d = n_patches, d_feat
        m = n_patches + d_feat

        cross_idx, cross_pi, cross_fi = [], [], []
        pp_idx, pp_i, pp_j = [], [], []
        ff_idx, ff_i, ff_j = [], [], []

        for idx, key in enumerate(output_keys):
            occ = list(key)
            pc, fc = sum(occ[:n_patches]), sum(occ[n_patches:])

            if pc == 1 and fc == 1:
                cross_idx.append(idx)
                cross_pi.append(next(i for i in range(n_patches) if occ[i] == 1))
                cross_fi.append(
                    next(i for i in range(n_patches, m) if occ[i] == 1) - n_patches
                )
            elif pc == 2 and fc == 0:
                modes = [i for i in range(n_patches) if occ[i] >= 1]
                if len(modes) == 2:
                    pp_idx.append(idx)
                    pp_i.append(modes[0])
                    pp_j.append(modes[1])
                elif len(modes) == 1:
                    pp_idx.append(idx)
                    pp_i.append(modes[0])
                    pp_j.append(modes[0])
            elif pc == 0 and fc == 2:
                modes = [i - n_patches for i in range(n_patches, m) if occ[i] >= 1]
                if len(modes) == 2:
                    ff_idx.append(idx)
                    ff_i.append(modes[0])
                    ff_j.append(modes[1])
                elif len(modes) == 1:
                    ff_idx.append(idx)
                    ff_i.append(modes[0])
                    ff_j.append(modes[0])

        for name, t in [
            ("cross_idx", cross_idx),
            ("cross_pi", cross_pi),
            ("cross_fi", cross_fi),
            ("pp_idx", pp_idx),
            ("pp_i", pp_i),
            ("pp_j", pp_j),
            ("ff_idx", ff_idx),
            ("ff_i", ff_i),
            ("ff_j", ff_j),
        ]:
            self.register_buffer(name, torch.tensor(t, dtype=torch.long))

    def forward(self, probs):
        bs = probs.shape[:-1]
        total = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        cv = probs[..., self.cross_idx]
        Y = torch.zeros(*bs, self.n, self.d, dtype=probs.dtype, device=probs.device)
        Y[..., self.cross_pi, self.cross_fi] = cv
        cm = cv.sum(dim=-1)
        Y = Y / cm.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-12)

        pv = probs[..., self.pp_idx]
        A = torch.zeros(*bs, self.n, self.n, dtype=probs.dtype, device=probs.device)
        A[..., self.pp_i, self.pp_j] = pv
        A[..., self.pp_j, self.pp_i] = (
            pv  # symmetrise (diagonal self-assigns harmlessly)
        )

        fv = probs[..., self.ff_idx]
        Fm = torch.zeros(*bs, self.d, self.d, dtype=probs.dtype, device=probs.device)
        Fm[..., self.ff_i, self.ff_j] = fv
        Fm[..., self.ff_j, self.ff_i] = fv

        masses = {
            "cross": (cm / total.squeeze(-1)).mean().item(),
            "pp": (pv.sum(-1) / total.squeeze(-1)).mean().item(),
            "ff": (fv.sum(-1) / total.squeeze(-1)).mean().item(),
        }
        return Y, A, Fm, masses


# ── TripleSectorReadout (3-photon, triple-cross + region-patch-patch) ──


class TripleSectorReadout(nn.Module):
    """
    Triple-cross (1r, 1p, 1f) → [r, p, d].
    Region-patch-patch (1r, 2p) → [r, p, p]  (hierarchical attention).
    """

    def __init__(
        self,
        n_regions: int,
        n_patches: int,
        d_feat: int,
        output_keys: list[tuple[int, ...]],
        extract_rpp: bool = True,
    ):
        super().__init__()
        self.r, self.p, self.d = n_regions, n_patches, d_feat
        r_end, p_end = n_regions, n_regions + n_patches
        m = n_regions + n_patches + d_feat
        self.extract_rpp = extract_rpp

        tc_idx, tc_ri, tc_pi, tc_fi = [], [], [], []
        rpp_idx, rpp_ri, rpp_pi, rpp_pj = [], [], [], []

        for idx, key in enumerate(output_keys):
            occ = list(key)
            rc, pc, fc = sum(occ[:r_end]), sum(occ[r_end:p_end]), sum(occ[p_end:])

            if rc == 1 and pc == 1 and fc == 1:
                tc_idx.append(idx)
                tc_ri.append(next(i for i in range(r_end) if occ[i] == 1))
                tc_pi.append(
                    next(i for i in range(r_end, p_end) if occ[i] == 1) - r_end
                )
                tc_fi.append(next(i for i in range(p_end, m) if occ[i] == 1) - p_end)

            elif extract_rpp and rc == 1 and pc == 2 and fc == 0:
                ri = next(i for i in range(r_end) if occ[i] == 1)
                pmodes = [i - r_end for i in range(r_end, p_end) if occ[i] >= 1]
                if len(pmodes) == 2:
                    rpp_idx.append(idx)
                    rpp_ri.append(ri)
                    rpp_pi.append(pmodes[0])
                    rpp_pj.append(pmodes[1])
                elif len(pmodes) == 1 and occ[r_end + pmodes[0]] == 2:
                    rpp_idx.append(idx)
                    rpp_ri.append(ri)
                    rpp_pi.append(pmodes[0])
                    rpp_pj.append(pmodes[0])

        for name, t in [
            ("tc_idx", tc_idx),
            ("tc_ri", tc_ri),
            ("tc_pi", tc_pi),
            ("tc_fi", tc_fi),
        ]:
            self.register_buffer(name, torch.tensor(t, dtype=torch.long))
        if extract_rpp:
            for name, t in [
                ("rpp_idx", rpp_idx),
                ("rpp_ri", rpp_ri),
                ("rpp_pi", rpp_pi),
                ("rpp_pj", rpp_pj),
            ]:
                self.register_buffer(name, torch.tensor(t, dtype=torch.long))

    def forward(self, probs):
        bs = probs.shape[:-1]
        total = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        tv = probs[..., self.tc_idx]
        T = torch.zeros(
            *bs, self.r, self.p, self.d, dtype=probs.dtype, device=probs.device
        )
        T[..., self.tc_ri, self.tc_pi, self.tc_fi] = tv
        tc_mass = tv.sum(dim=-1)
        T = T / tc_mass.view(*bs, 1, 1, 1).clamp(min=1e-12)
        masses = {"triple_cross": (tc_mass / total.squeeze(-1)).mean().item()}

        A_rpp = None
        if self.extract_rpp and hasattr(self, "rpp_idx") and self.rpp_idx.numel() > 0:
            rv = probs[..., self.rpp_idx]
            A_rpp = torch.zeros(
                *bs, self.r, self.p, self.p, dtype=probs.dtype, device=probs.device
            )
            A_rpp[..., self.rpp_ri, self.rpp_pi, self.rpp_pj] = rv
            A_rpp[..., self.rpp_ri, self.rpp_pj, self.rpp_pi] = rv
            masses["rpp"] = (rv.sum(-1) / total.squeeze(-1)).mean().item()

        return T, A_rpp, masses
