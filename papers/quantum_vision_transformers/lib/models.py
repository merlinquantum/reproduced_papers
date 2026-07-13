"""
Quantum Vision Transformer architectures — photonic reproduction.

Paper models (Cherrat et al.):
  A. Orthogonal Patch-wise NN       1 photon, d modes, shared V
  B. Quantum Orthogonal Transformer 1 photon, d modes, V + W, overlap attention
  C. Direct Quantum Attention       1 photon, d modes, pragmatic C2 route
  D. Compound Transformer           2 photons, (n+d) modes, cross-partition readout

Extensions beyond the paper:
  D (full_sector)  — use all three 2-photon sectors (cross + pp + ff)
  E. Multi-sector  — shared circuit, 1-photon features + 2-photon attention
  F. Hierarchical  — 3 photons, region + patch + feature hierarchy

Paper baselines:
  VisionTransformer — classical baseline from Appendix A
  OrthoFNN          — quantum baseline without attention
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import merlin as ML
from merlin.builder import CircuitBuilder
from merlin.measurement import MeasurementStrategy
from merlin.core.computation_space import ComputationSpace
from merlin.core.state_vector import StateVector

from .photonic_primitives import (
    TrainableInterferometer,
    OverlapEstimator,
    CompoundSectorReadout,
    FullSectorReadout,
    TripleSectorReadout,
    complex_dtype_for,
    normalize_for_encoding,
    _fock_basis_size,
)
from .data import ClassicalPatchEmbed, HierarchicalPatchEmbed, ImageLinearEmbed


# ── helpers ─────────────────────────────────────────────────────────────

def _classify_output_keys(output_keys, n_block1, n_block2=0):
    """
    Classify 2-photon (or 3-photon) Fock basis states by which mode-blocks
    the photons land in.  Returns dict mapping (block_counts) → list of
    (basis_idx, mode_indices_within_blocks).

    For 2-photon with blocks [patch(n), feat(d)]:
        (1,1) → cross-partition entries as (idx, patch_i, feat_j)
    """
    result = {}
    for idx, key in enumerate(output_keys):
        occ = list(key)
        result[idx] = occ
    return result


def _build_encoding_indices(output_keys, n_patches, d, n_photons=2):
    """
    Pre-compute vectorised encoding tensors for the cross-partition sector.
    Returns (basis_indices, data_row_indices, data_col_indices) as LongTensors.
    """
    basis_idx, row_idx, col_idx = [], [], []
    total = n_patches + d
    for idx, key in enumerate(output_keys):
        occ = list(key)
        pc = sum(occ[:n_patches])
        fc = sum(occ[n_patches:total])
        if pc == 1 and fc == 1 and pc + fc == n_photons:
            pi = next(i for i in range(n_patches) if occ[i] == 1)
            fi = next(i for i in range(n_patches, total) if occ[i] == 1) - n_patches
            basis_idx.append(idx)
            row_idx.append(pi)
            col_idx.append(fi)
    return (torch.tensor(basis_idx, dtype=torch.long),
            torch.tensor(row_idx, dtype=torch.long),
            torch.tensor(col_idx, dtype=torch.long))


def _build_triple_encoding_indices(output_keys, r, p, d):
    """Pre-compute vectorised encoding for the triple-cross sector (3 photons)."""
    r_end = r
    p_end = r + p
    basis_idx, ri_list, pi_list, fi_list = [], [], [], []
    for idx, key in enumerate(output_keys):
        occ = list(key)
        rc = sum(occ[:r_end])
        pc = sum(occ[r_end:p_end])
        fc = sum(occ[p_end:])
        if rc == 1 and pc == 1 and fc == 1:
            ri = next(i for i in range(r_end) if occ[i] == 1)
            pi = next(i for i in range(r_end, p_end) if occ[i] == 1) - r_end
            fi = next(i for i in range(p_end, r + p + d) if occ[i] == 1) - p_end
            basis_idx.append(idx)
            ri_list.append(ri)
            pi_list.append(pi)
            fi_list.append(fi)
    return (torch.tensor(basis_idx, dtype=torch.long),
            torch.tensor(ri_list, dtype=torch.long),
            torch.tensor(pi_list, dtype=torch.long),
            torch.tensor(fi_list, dtype=torch.long))


def _forward_single_photon_probs(layer, x: torch.Tensor) -> torch.Tensor:
    """Apply a 1-photon layer to [..., d] inputs and return probabilities with the same shape."""
    x_enc = normalize_for_encoding(x)
    shape, dtype = x_enc.shape, x_enc.dtype

    if hasattr(layer, "forward_tensor"):
        return layer.forward_tensor(x_enc)

    flat = x_enc.reshape(-1, shape[-1])
    sv = StateVector.from_tensor(
        flat.to(complex_dtype_for(x_enc.dtype)),
        n_modes=shape[-1],
        n_photons=1,
    )
    result = layer(sv)
    amps_or_probs = result.to_dense() if isinstance(result, StateVector) else result

    if torch.is_complex(amps_or_probs):
        amps_or_probs = amps_or_probs.real.pow(2) + amps_or_probs.imag.pow(2)

    return amps_or_probs.to(dtype).reshape(shape)


def _forward_single_photon_amplitudes(layer, x: torch.Tensor) -> torch.Tensor:
    """Apply a 1-photon layer to [..., d] inputs and return complex amplitudes with the same shape."""
    x_enc = normalize_for_encoding(x)
    shape = x_enc.shape

    if hasattr(layer, "forward_complex"):
        return layer.forward_complex(x_enc)

    flat = x_enc.reshape(-1, shape[-1])
    sv = StateVector.from_tensor(
        flat.to(complex_dtype_for(x_enc.dtype)),
        n_modes=shape[-1],
        n_photons=1,
    )
    result = layer(sv)
    amps = result.to_dense() if isinstance(result, StateVector) else result
    return amps.reshape(*shape[:-1], shape[-1])


def _overlap_scores(layer, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """Compute |<x_i | W | x_j>|^2 for [..., n, d] inputs under either generic or butterfly layers."""
    Wx_j = _forward_single_photon_amplitudes(layer, x_j)
    overlap = torch.einsum("...id,...jd->...ij", normalize_for_encoding(x_i).to(Wx_j.dtype), Wx_j)
    return overlap.abs().pow(2).to(x_i.dtype)


# ── classical head ──────────────────────────────────────────────────────

class ClassicalHead(nn.Module):
    def __init__(self, d: int, n_classes: int, use_cls_token: bool = True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return self.fc(feat)


class ClassicalVisionAttention(nn.Module):
    """Classical attention block matching the simplified paper baseline in Appendix A."""

    def __init__(self, d: int):
        super().__init__()
        self.V = nn.Linear(d, d, bias=False)
        self.W = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.V(x)
        scores = torch.einsum("bnd,df,bmf->bnm", x, self.W, x)
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bnm,bmd->bnd", attn, features)


# ═══════════════════════════════════════════════════════════════════════
# Paper models
# ═══════════════════════════════════════════════════════════════════════

class ModelA(nn.Module):
    """Orthogonal Patch-wise: y_i = V x_i.  1 photon, d modes."""
    def __init__(self, d: int, circuit_family: str = "generic", device=None):
        super().__init__()
        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            circuit = make_butterfly_mzi_circuit(d, prefix="V")
            self.V = ML.QuantumLayer(
                circuit=circuit, n_photons=1,
                trainable_parameters=["V"],
                measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
                device=device
            )
        else:
            self.V = TrainableInterferometer(d, n_photons=1, name="V", device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward_single_photon_probs(self.V, x)


class ModelB(nn.Module):
    """Quantum Orthogonal Transformer.  V (features) + W (attention)."""
    def __init__(self, d: int, circuit_family: str = "generic", device=None):
        super().__init__()
        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            self.V = ML.QuantumLayer(
                circuit=make_butterfly_mzi_circuit(d, prefix="V"),
                n_photons=1, trainable_parameters=["V"],
                measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
                device=device
            )
            self.W = ML.QuantumLayer(
                circuit=make_butterfly_mzi_circuit(d, prefix="W"),
                n_photons=1, trainable_parameters=["W"],
                measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
                return_object=True,
                device=device
            )
        else:
            self.V = TrainableInterferometer(d, n_photons=1, name="V", device=device)
            self.W = TrainableInterferometer(d, n_photons=1, name="W", device=device)
        self.overlap = OverlapEstimator(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = normalize_for_encoding(x)
        z = _forward_single_photon_probs(self.V, x_enc)
        A_scores = self.overlap(x_enc, x_enc) if hasattr(self.W, "forward_complex") else _overlap_scores(self.W, x_enc, x_enc)
        A = F.softmax(A_scores, dim=-1)
        return torch.einsum("...ij,...jd->...id", A, z)


class ModelC(nn.Module):
    """Direct Quantum Attention (pragmatic hybrid C2)."""
    def __init__(self, d: int, circuit_family: str = "generic", device=None):
        super().__init__()
        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            self.V = ML.QuantumLayer(
                circuit=make_butterfly_mzi_circuit(d, prefix="V"),
                n_photons=1, trainable_parameters=["V"],
                measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
                device=device
            )
            self.W = ML.QuantumLayer(
                circuit=make_butterfly_mzi_circuit(d, prefix="W"),
                n_photons=1, trainable_parameters=["W"],
                measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
                return_object=True,
                device=device
            )
        else:
            self.V = TrainableInterferometer(d, n_photons=1, name="V", device=device)
            self.W = TrainableInterferometer(d, n_photons=1, name="W", device=device)
        self.overlap = OverlapEstimator(self.W)

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None):
        x_enc = normalize_for_encoding(x)
        if A is None:
            A_scores = self.overlap(x_enc, x_enc) if hasattr(self.W, "forward_complex") else _overlap_scores(self.W, x_enc, x_enc)
            A = F.softmax(A_scores, dim=-1)
        weighted = normalize_for_encoding(
            torch.einsum("...ij,...jd->...id", A, x_enc)
        )
        return _forward_single_photon_probs(self.V, weighted)


# ═══════════════════════════════════════════════════════════════════════
# Model D — Compound Transformer (2 photons)
# ═══════════════════════════════════════════════════════════════════════

class CompoundTransformerLayer(nn.Module):
    """
    2-photon compound layer.  One (n+d)-mode interferometer.

    compound_readout:
      "cross_only"  — paper default (only cross-partition → [n, d])
      "full_sector" — extension (cross + patch-patch + feat-feat)
    """

    def __init__(self, n_patches: int, d: int,
                 compound_readout: str = "cross_only",
                 circuit_family: str = "generic", device=None):
        super().__init__()
        self.n = n_patches
        self.d = d
        self.total_modes = n_patches + d
        self.compound_readout = compound_readout

        # Probability readout — native photonic measurement (photon counting).
        # merlin 0.4 removed partition_blocks/allowed_counts from
        # MeasurementStrategy.probs; the layer now returns the full Fock
        # distribution and sector selection happens classically in the
        # readout modules (CompoundSectorReadout / FullSectorReadout).
        measurement_strategy = MeasurementStrategy.probs(ComputationSpace.FOCK)
        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            circuit = make_butterfly_mzi_circuit(self.total_modes, prefix="Vc")
            self.layer = ML.QuantumLayer(
                circuit=circuit, n_photons=2,
                trainable_parameters=["Vc"],
                measurement_strategy=measurement_strategy,
                device=device
            )
        else:
            builder = CircuitBuilder(n_modes=self.total_modes)
            builder.add_entangling_layer(trainable=True, model="mzi", name="Vc")
            self.layer = ML.QuantumLayer(
                builder=builder, n_photons=2,
                measurement_strategy=measurement_strategy,
                device=device,
            )

        output_keys = list(self.layer.output_keys)
        self.basis_size = len(output_keys)

        if compound_readout == "full_sector":
            self.readout = FullSectorReadout(n_patches, d, output_keys)
            self.ff_proj = nn.Linear(d, d, bias=False)
        else:
            self.readout = CompoundSectorReadout(n_patches, d, output_keys)

        # vectorised encoding: pre-computed index tensors
        bi, ri, ci = _build_encoding_indices(output_keys, n_patches, d)
        self.register_buffer("_enc_basis", bi)
        self.register_buffer("_enc_row", ri)
        self.register_buffer("_enc_col", ci)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, n, d] → 2-photon amplitude tensor aligned with layer.output_keys."""
        B = X.shape[0]
        complex_dtype = complex_dtype_for(X.dtype)
        amps = torch.zeros(B, self.basis_size, dtype=complex_dtype, device=X.device)
        amps[:, self._enc_basis] = X[:, self._enc_row, self._enc_col].to(complex_dtype)
        norm = amps.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        return amps / norm

    def forward(self, X: torch.Tensor):
        sv = self._encode(X)
        probs = self.layer(sv).to(X.dtype)

        if self.compound_readout == "full_sector":
            Y_cross, A_pp, F_ff, masses = self.readout(probs)
            A = F.softmax(A_pp, dim=-1)
            Y = torch.einsum("...ij,...jd->...id", A, Y_cross)
            ff_diag = torch.diagonal(F_ff, dim1=-2, dim2=-1)
            Y = Y + self.ff_proj(ff_diag).unsqueeze(-2)
            info = {"sector_masses": masses, "A_pp": A_pp.detach()}
            return Y, info
        else:
            return self.readout(probs)


class ModelD(nn.Module):
    """Stack of CompoundTransformerLayers."""
    def __init__(self, n_patches: int, d: int, n_layers: int = 1,
                 compound_readout: str = "cross_only",
                 circuit_family: str = "generic", device=None):
        super().__init__()
        self.layers = nn.ModuleList([
            CompoundTransformerLayer(n_patches, d, compound_readout,
                                     circuit_family=circuit_family, device=device)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        sector_masses = []
        for layer in self.layers:
            x, sm = layer(x)
            sector_masses.append(sm)
        return x, sector_masses


# ═══════════════════════════════════════════════════════════════════════
# Model E — Multi-sector attention (shared circuit, 1ph + 2ph)
# ═══════════════════════════════════════════════════════════════════════

class MultiSectorLayer(nn.Module):
    """
    One shared (n+d)-mode interferometer.  Two photon-number readouts:
      1-photon → per-patch feature transform  (replaces V)
      2-photon → emergent attention from patch-patch sector  (replaces W)

    Parameter tying: both QuantumLayers are built from the same Perceval
    circuit.  After construction, layer_2ph's nn.Parameters are replaced
    with references to layer_1ph's so gradients flow through one set of
    angles.  If MerLin changes its internal parameter naming, this tying
    may need updating — check by verifying layer_2ph has zero independent
    parameters after init.
    """

    def __init__(self, n_patches: int, d: int,
                 circuit_family: str = "generic", device=None):
        super().__init__()
        self.n = n_patches
        self.d = d
        m = n_patches + d
        self.total_modes = m

        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            pcvl_circuit = make_butterfly_mzi_circuit(m, prefix="shared")
            tp = ["shared"]
        else:
            builder = CircuitBuilder(n_modes=m)
            builder.add_entangling_layer(trainable=True, model="mzi", name="shared")
            pcvl_circuit = builder.to_pcvl_circuit()
            tp = builder.trainable_parameter_prefixes

        self.layer_1ph = ML.QuantumLayer(
            circuit=pcvl_circuit, n_photons=1, trainable_parameters=tp,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
            device=device,
        )
        self.layer_2ph = ML.QuantumLayer(
            circuit=pcvl_circuit, n_photons=2, trainable_parameters=tp,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
            device=device,
        )

        # tie parameters
        for prefix in tp:
            shared = getattr(self.layer_1ph, prefix, None)
            if shared is not None and isinstance(shared, nn.Parameter):
                self.layer_2ph.register_parameter(prefix, shared)
        self.layer_2ph.thetas = list(self.layer_1ph.thetas)

        # 2ph patch-patch sector indices
        keys_2ph = list(self.layer_2ph.output_keys)
        pp_idx, pp_i, pp_j = [], [], []
        for idx, key in enumerate(keys_2ph):
            occ = list(key)
            pc, fc = sum(occ[:n_patches]), sum(occ[n_patches:])
            if pc == 2 and fc == 0:
                modes = [i for i in range(n_patches) if occ[i] >= 1]
                if len(modes) == 2:
                    pp_idx.append(idx); pp_i.append(modes[0]); pp_j.append(modes[1])
                elif len(modes) == 1 and occ[modes[0]] == 2:
                    pp_idx.append(idx); pp_i.append(modes[0]); pp_j.append(modes[0])
        self.register_buffer("pp_idx", torch.tensor(pp_idx, dtype=torch.long))
        self.register_buffer("pp_i", torch.tensor(pp_i, dtype=torch.long))
        self.register_buffer("pp_j", torch.tensor(pp_j, dtype=torch.long))

        # 2ph cross-partition indices (for encoding)
        bi, ri, ci = _build_encoding_indices(keys_2ph, n_patches, d)
        self.register_buffer("_enc_basis", bi)
        self.register_buffer("_enc_row", ri)
        self.register_buffer("_enc_col", ci)
        self.basis_size_2ph = len(keys_2ph)

    def _feature_transform(self, x: torch.Tensor) -> torch.Tensor:
        """1ph path: [B, n, d] → [B, n, d] feature probabilities.
        For 1 photon over m modes, Fock basis = modes, so probs[j] = |α_j|²."""
        B, n, d = x.shape
        m = self.total_modes
        padded = torch.zeros(B * n, m, dtype=x.dtype, device=x.device)
        padded[:, self.n:] = x.reshape(B * n, d)
        padded = normalize_for_encoding(padded)
        sv = StateVector.from_tensor(
            padded.to(complex_dtype_for(x.dtype)),
            n_modes=m,
            n_photons=1,
        )
        probs = self.layer_1ph(sv).to(x.dtype)              # [B*n, m]
        return probs[:, self.n:].reshape(B, n, d)            # feature modes only

    def _encode_2ph(self, X: torch.Tensor) -> StateVector:
        """[B, n, d] → 2-photon StateVector (vectorised)."""
        B = X.shape[0]
        complex_dtype = complex_dtype_for(X.dtype)
        amps = torch.zeros(B, self.basis_size_2ph, dtype=complex_dtype, device=X.device)
        amps[:, self._enc_basis] = X[:, self._enc_row, self._enc_col].to(complex_dtype)
        norm = amps.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        return StateVector.from_tensor(amps / norm, n_modes=self.total_modes, n_photons=2)

    def forward(self, X: torch.Tensor):
        z = self._feature_transform(X)
        sv2 = self._encode_2ph(X)
        probs = self.layer_2ph(sv2).to(X.dtype)
        # patch-patch attention
        bs = probs.shape[:-1]
        pv = probs[..., self.pp_idx]
        A_pp = torch.zeros(*bs, self.n, self.n, dtype=probs.dtype, device=probs.device)
        A_pp[..., self.pp_i, self.pp_j] = pv
        A_pp[..., self.pp_j, self.pp_i] = pv
        A = F.softmax(A_pp, dim=-1)
        Y = torch.einsum("...ij,...jd->...id", A, z)
        total = probs.sum(dim=-1).clamp(min=1e-12)
        info = {"sector_masses": {"pp": (pv.sum(dim=-1) / total).mean().item()},
                "A_pp": A_pp.detach()}
        return Y, info


# ═══════════════════════════════════════════════════════════════════════
# Model F — Hierarchical 3-photon compound
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalCompoundLayer(nn.Module):
    """
    3-photon compound over [region (r) | patch (p) | feature (d)] modes.

    V^(3) jointly mixes across all three hierarchies.  The region-patch-
    patch sector (1r, 2p) provides per-region hierarchical attention.
    """

    def __init__(self, n_regions: int, n_patches_per_region: int, d: int,
                 use_rpp_attention: bool = True,
                 circuit_family: str = "generic", device=None):
        super().__init__()
        self.r = n_regions
        self.p = n_patches_per_region
        self.d = d
        self.total_modes = n_regions + n_patches_per_region + d
        self.use_rpp_attention = use_rpp_attention

        # merlin 0.4 removed partition_blocks/allowed_counts; the layer returns
        # the full 3-photon Fock distribution and TripleSectorReadout selects
        # the (1,1,1) / (1,2,0) sectors classically.
        measurement_strategy = MeasurementStrategy.probs(ComputationSpace.FOCK)
        if circuit_family == "butterfly":
            from .structured_circuits import make_butterfly_mzi_circuit
            circuit = make_butterfly_mzi_circuit(self.total_modes, prefix="Vh")
            self.layer = ML.QuantumLayer(
                circuit=circuit, n_photons=3,
                trainable_parameters=["Vh"],
                measurement_strategy=measurement_strategy,
                device=device
            )
        else:
            builder = CircuitBuilder(n_modes=self.total_modes)
            builder.add_entangling_layer(trainable=True, model="mzi", name="Vh")
            self.layer = ML.QuantumLayer(
                builder=builder, n_photons=3,
                measurement_strategy=measurement_strategy,
                device=device,
            )

        output_keys = list(self.layer.output_keys)
        self.basis_size = len(output_keys)
        self.readout = TripleSectorReadout(
            n_regions, n_patches_per_region, d, output_keys,
            extract_rpp=use_rpp_attention,
        )

        # vectorised encoding
        bi, ri, pi, fi = _build_triple_encoding_indices(
            output_keys, n_regions, n_patches_per_region, d
        )
        self.register_buffer("_enc_basis", bi)
        self.register_buffer("_enc_ri", ri)
        self.register_buffer("_enc_pi", pi)
        self.register_buffer("_enc_fi", fi)

    def _encode(self, T: torch.Tensor) -> torch.Tensor:
        """T: [B, r, p, d] → 3-photon amplitude tensor aligned with layer.output_keys."""
        B = T.shape[0]
        complex_dtype = complex_dtype_for(T.dtype)
        amps = torch.zeros(B, self.basis_size, dtype=complex_dtype, device=T.device)
        amps[:, self._enc_basis] = T[:, self._enc_ri, self._enc_pi, self._enc_fi].to(complex_dtype)
        norm = amps.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        return amps / norm

    def forward(self, x: torch.Tensor):
        """x: [B, n, d] (n = r*p) → (Y: [B, n, d], info)."""
        B = x.shape[0]
        T = x.reshape(B, self.r, self.p, self.d)
        sv = self._encode(T)
        probs = self.layer(sv).to(x.dtype)

        Y_rpd, A_rpp, masses = self.readout(probs)

        if self.use_rpp_attention and A_rpp is not None:
            A = F.softmax(A_rpp, dim=-1)
            Y_rpd = torch.einsum("...rpq,...rqd->...rpd", A, Y_rpd)

        Y = Y_rpd.reshape(B, self.r * self.p, self.d)
        info = {"sector_masses": masses}
        if A_rpp is not None:
            info["A_rpp"] = A_rpp.detach()
        return Y, info


class OrthoFNNModel(nn.Module):
    """Quantum orthogonal fully connected baseline: one global image embedding, no attention."""

    def __init__(self, d: int, n_layers: int = 4, circuit_family: str = "generic", device=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [ModelA(d, circuit_family=circuit_family, device=device) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Full QVT Model
# ═══════════════════════════════════════════════════════════════════════

class QVTModel(nn.Module):
    """
    image → patch embed → [cls + pos] → L × (norm → attn → residual → MLP) → head

    model_type: A, B, C, D, E, F, VisionTransformer, OrthoFNN
    compound_readout: "cross_only" | "full_sector"  (D only)
    """

    def __init__(
        self,
        model_type: str = "B",
        img_size: int = 28,
        in_channels: int = 3,
        patch_size: int = 7,
        embed_dim: int = 16,
        n_layers: int = 4,
        n_classes: int = 5,
        use_cls_token: bool = True,
        use_pos_embed: bool = True,
        image_embed_grayscale: bool = False,
        compound_readout: str = "cross_only",
        circuit_family: str = "generic",
        n_regions_per_side: int = 2,
        n_patches_per_side: int = 2,
        use_rpp_attention: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.model_type = model_type

        # ── power-of-two check for butterfly ──
        if circuit_family == "butterfly":
            from .structured_circuits import _is_power_of_two
            # For A/B/C, d must be power of 2
            if model_type in ("A", "B", "C", "OrthoFNN"):
                if not _is_power_of_two(embed_dim):
                    raise ValueError(f"Butterfly path for Model {model_type} requires embed_dim={embed_dim} to be power of 2.")
            # For D, E, (n+d) must be power of 2
            # For D, E, we recommended disabling CLS for 32 modes if d=16, n=16.
            # But here we just check the final total sequence length + d.
            # (Note: we also check it later once total_seq is known)

        # ── patch embedding ──
        if model_type == "F":
            self.patch_embed = HierarchicalPatchEmbed(
                img_size=img_size, in_channels=in_channels,
                n_regions_per_side=n_regions_per_side,
                n_patches_per_side=n_patches_per_side,
                embed_dim=embed_dim,
            )
            n_patches = self.patch_embed.n_patches
            n_regions = self.patch_embed.n_regions
            n_pp_region = self.patch_embed.n_patches_per_region
            use_cls_token = False  # F requires n = r*p exactly
        elif model_type == "OrthoFNN":
            self.patch_embed = ImageLinearEmbed(
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                grayscale=image_embed_grayscale,
            )
            n_patches = 1
            use_cls_token = False
            use_pos_embed = False
        else:
            self.patch_embed = ClassicalPatchEmbed(
                img_size=img_size, in_channels=in_channels,
                patch_size=patch_size, embed_dim=embed_dim,
            )
            n_patches = self.patch_embed.n_patches

        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            total_seq = n_patches + 1
        else:
            total_seq = n_patches
            total_seq = n_patches

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, total_seq, embed_dim) * 0.02)

        # ── power-of-two check (continued) ──
        if circuit_family == "butterfly":
            from .structured_circuits import _is_power_of_two
            if model_type in ("D", "E"):
                total_modes = total_seq + embed_dim
                if not _is_power_of_two(total_modes):
                    raise ValueError(
                        f"Butterfly path for Model {model_type} requires total_modes={total_modes} "
                        f"to be power of 2. (Current: n_patches={n_patches}, d={embed_dim}, "
                        f"use_cls_token={use_cls_token} -> total={total_modes}). "
                        f"Try disabling CLS token if total is 33."
                    )
            if model_type == "F":
                total_modes = n_regions + n_pp_region + embed_dim
                if not _is_power_of_two(total_modes):
                    raise ValueError(f"Butterfly path for Model F requires total_modes={total_modes} to be power of 2.")

        # ── attention layers (device passed to QuantumLayer constructors) ──
        self.attn_layers = nn.ModuleList()
        if model_type == "OrthoFNN":
            # Match the dedicated baseline semantics: global image embedding, then
            # repeated orthogonal patch-wise layers, without transformer residual/MLP blocks.
            self.attn_layers = nn.ModuleList([
                ModelA(embed_dim, circuit_family=circuit_family, device=device)
                for _ in range(n_layers)
            ])
            self.pre_norms = nn.ModuleList()
            self.post_norms = nn.ModuleList()
            self.mlps = nn.ModuleList()
        else:
            for _ in range(n_layers):
                if model_type == "A":
                    self.attn_layers.append(ModelA(embed_dim, circuit_family=circuit_family, device=device))
                elif model_type == "B":
                    self.attn_layers.append(ModelB(embed_dim, circuit_family=circuit_family, device=device))
                elif model_type == "C":
                    self.attn_layers.append(ModelC(embed_dim, circuit_family=circuit_family, device=device))
                elif model_type == "VisionTransformer":
                    self.attn_layers.append(ClassicalVisionAttention(embed_dim))
                elif model_type == "D":
                    self.attn_layers.append(
                        CompoundTransformerLayer(total_seq, embed_dim, compound_readout,
                                                circuit_family=circuit_family, device=device))
                elif model_type == "E":
                    self.attn_layers.append(MultiSectorLayer(total_seq, embed_dim,
                                                             circuit_family=circuit_family, device=device))
                elif model_type == "F":
                    self.attn_layers.append(
                        HierarchicalCompoundLayer(n_regions, n_pp_region, embed_dim,
                                                  use_rpp_attention, circuit_family=circuit_family, device=device))
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

            # ── per-layer norms and MLPs ──
            self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
            self.post_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                ) for _ in range(n_layers)
            ])
        self.head = ClassicalHead(embed_dim, n_classes, use_cls_token)
        self.sector_masses: list = []

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        """Subtract per-vector mean.  Converts non-negative probabilities to
        signed deviations.  Zero learnable parameters — all discriminative
        power must come from the quantum circuit or MLP.
        For a probability vector (sums to 1), this equals p - 1/d."""
        return x - x.mean(dim=-1, keepdim=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        if self.model_type == "OrthoFNN":
            for layer in self.attn_layers:
                x = layer(x)
            return self.head(x)

        if x.ndim == 4:  # HierarchicalPatchEmbed → flatten [B,r,p,d] → [B,n,d]
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if self.use_pos_embed:
            x = x + self.pos_embed

        self.sector_masses = []
        for i, layer in enumerate(self.attn_layers):
            h = normalize_for_encoding(self.pre_norms[i](x))
            if self.model_type in ("D", "E", "F"):
                attn_out, sm = layer(h)
                self.sector_masses.append(sm)
            else:
                attn_out = layer(h)
            # Photonic paths emit non-negative probabilities and need centering before the residual path.
            if self.model_type != "VisionTransformer":
                attn_out = self._center(attn_out)
            x = x + attn_out
            x = x + self.mlps[i](self.post_norms[i](x))

        return self.head(x)

    def count_trainable_params(self) -> dict:
        c = {}
        c["patch_embed"] = sum(p.numel() for p in self.patch_embed.parameters() if p.requires_grad)
        c["attention"] = sum(p.numel() for n, p in self.named_parameters()
                             if p.requires_grad and "attn_layers." in n)
        c["mlp"] = sum(p.numel() for n, p in self.named_parameters()
                       if p.requires_grad and "mlps." in n)
        c["head"] = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        c["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return c
