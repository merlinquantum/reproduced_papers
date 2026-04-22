#!/usr/bin/env python3
"""
Verification checklist for QVT — tests all models A–F.
Run: python verify.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn.functional as F
torch.set_default_dtype(torch.float64)

from lib.photonic_primitives import TrainableInterferometer, OverlapEstimator, normalize_for_encoding
from lib.models import ModelA, ModelB, ModelC, CompoundTransformerLayer, MultiSectorLayer, HierarchicalCompoundLayer, QVTModel
from lib.data import ClassicalPatchEmbed, HierarchicalPatchEmbed
from lib.structured_circuits import butterfly_spec, butterfly_param_count, make_butterfly_mzi_circuit

PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
results = []

def check(name, cond):
    results.append((name, cond))
    print(f"  {PASS if cond else FAIL} {name}")

def ones(t, *shape):
    return torch.ones(*shape, dtype=t.dtype, device=t.device)

print("=" * 70)
print("QVT Photonic Reproduction — Full Verification")
print("=" * 70)

# ── Model A ─────────────────────────────────────────────────────────────
print("\n— A: Orthogonal Patch-wise —")
V = TrainableInterferometer(4, n_photons=1, name="Vt")
x = normalize_for_encoding(torch.randn(3, 4))
y = V.forward_tensor(x)
check("State prep unit norm", torch.allclose(x.norm(dim=-1), ones(x, 3), atol=1e-6))
check("Output shape", y.shape == x.shape)
check("Output sums to ≈ 1 (probability conservation)",
      torch.allclose(y.sum(dim=-1), ones(y, 3), atol=1e-3))
y.sum().backward()
check("Gradients flow", any(p.grad is not None and p.grad.abs().sum() > 0 for p in V.parameters()))

# ── Model B ─────────────────────────────────────────────────────────────
print("\n— B: Quantum Orthogonal Transformer —")
b = ModelB(4)
xb = normalize_for_encoding(torch.randn(2, 5, 4))
yb = b(xb)
check("Output shape", yb.shape == xb.shape)
A_raw = b.overlap(normalize_for_encoding(xb), normalize_for_encoding(xb))
check("Attention non-negative", (A_raw >= -1e-10).all().item())
A_soft = F.softmax(A_raw, dim=-1)
check("Softmax sums to 1", torch.allclose(A_soft.sum(-1), ones(A_soft, 2, 5), atol=1e-5))

# ── Model C ─────────────────────────────────────────────────────────────
print("\n— C: Direct Quantum Attention —")
c = ModelC(4)
yc = c(xb)
check("Output shape", yc.shape == xb.shape)

# ── Model D (cross_only) ───────────────────────────────────────────────
print("\n— D: Compound (cross_only) —")
ld = CompoundTransformerLayer(3, 4, "cross_only")
Xd = normalize_for_encoding(torch.randn(2, 3, 4))
Yd, sm = ld(Xd)
check("Output shape [B,n,d]", Yd.shape == (2, 3, 4))
check(f"Sector mass {sm.mean():.3f} > 0.05", sm.mean().item() > 0.05)
Yd[:, 0, 0].mean().backward()
check("Gradients flow", any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in ld.parameters() if p.requires_grad))

# ── Model D (full_sector) ──────────────────────────────────────────────
print("\n— D: Compound (full_sector) —")
ldf = CompoundTransformerLayer(3, 4, "full_sector")
Xdf = normalize_for_encoding(torch.randn(2, 3, 4))
Ydf, info_df = ldf(Xdf)
check("Output shape", Ydf.shape == (2, 3, 4))
check("Has sector_masses dict", isinstance(info_df, dict) and "sector_masses" in info_df)
masses = info_df["sector_masses"]
check("Three sector fractions", all(k in masses for k in ("cross", "pp", "ff")))

# ── Model E ─────────────────────────────────────────────────────────────
print("\n— E: Multi-sector attention —")
le = MultiSectorLayer(3, 4)
Xe = normalize_for_encoding(torch.randn(2, 3, 4))
Ye, info_e = le(Xe)
check("Output shape", Ye.shape == (2, 3, 4))
check("Has pp sector mass", "pp" in info_e.get("sector_masses", {}))
# check parameter sharing: layer_2ph should have no independent params
shared_ids = {id(p) for p in le.layer_1ph.parameters()}
independent_2ph = [p for p in le.layer_2ph.parameters() if id(p) not in shared_ids]
check(f"Params tied ({len(independent_2ph)} independent in 2ph)", len(independent_2ph) == 0)

# ── Model F ─────────────────────────────────────────────────────────────
print("\n— F: Hierarchical 3-photon —")
lf = HierarchicalCompoundLayer(n_regions=2, n_patches_per_region=2, d=4)
Xf = normalize_for_encoding(torch.randn(2, 4, 4))  # n = r*p = 4
Yf, info_f = lf(Xf)
check("Output shape [B,n,d]", Yf.shape == (2, 4, 4))
check("Has triple_cross mass", "triple_cross" in info_f.get("sector_masses", {}))
Yf[:, 0, 0].mean().backward()
check("Gradients flow", any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in lf.parameters() if p.requires_grad))

# ── HierarchicalPatchEmbed ──────────────────────────────────────────────
print("\n— HierarchicalPatchEmbed —")
hpe = HierarchicalPatchEmbed(img_size=28, in_channels=3,
                              n_regions_per_side=2, n_patches_per_side=2, embed_dim=8)
imgs = torch.randn(2, 3, 28, 28)
out = hpe(imgs)
check(f"Shape [B,r,p,d] = {list(out.shape)}", out.shape == (2, 4, 4, 8))
check("Normalised", torch.allclose(out.norm(dim=-1), ones(out, 2, 4, 4), atol=1e-6))

# ── Structured butterfly circuit ────────────────────────────────────────
print("\n— Structured butterfly circuit —")
spec = butterfly_spec(8)
check("Butterfly stages = log2(n)", spec.n_stages == 3)
check("Butterfly pairing schedule matches radix-2 layout", spec.pairings == [
    [(0, 1), (2, 3), (4, 5), (6, 7)],
    [(0, 2), (1, 3), (4, 6), (5, 7)],
    [(0, 4), (1, 5), (2, 6), (3, 7)],
])
butterfly = make_butterfly_mzi_circuit(8, prefix="T")
check("Butterfly circuit builds", butterfly.m == 8)
check("Butterfly parameter count matches schedule",
      len(butterfly.get_parameters()) == butterfly_param_count(8))

# ── Baselines ──────────────────────────────────────────────────────────
print("\n— Paper baselines —")
imgs = torch.randn(2, 3, 28, 28)
for mt in ("VisionTransformer", "OrthoFNN"):
    m = QVTModel(model_type=mt, n_classes=5, n_layers=1, embed_dim=8)
    logits = m(imgs)
    loss = F.cross_entropy(logits, torch.randint(0, 5, (2,)))
    loss.backward()
    check(f"Baseline {mt}: fwd+bwd OK", logits.shape == (2, 5))

# ── Full pipeline (A, B, C) ────────────────────────────────────────────
print("\n— Full QVT integration (A, B, C) —")
imgs = torch.randn(2, 3, 28, 28)
for mt in ("A", "B", "C"):
    m = QVTModel(model_type=mt, n_classes=5, n_layers=1, embed_dim=8)
    logits = m(imgs)
    loss = F.cross_entropy(logits, torch.randint(0, 5, (2,)))
    loss.backward()
    pc = m.count_trainable_params()
    check(f"Model {mt}: fwd+bwd OK (attn={pc['attention']})", logits.shape == (2, 5))

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
p = sum(c for _, c in results)
print(f"{p}/{len(results)} checks passed" +
      (" — all OK!" if p == len(results) else " — FAILURES:"))
for name, c in results:
    if not c:
        print(f"  {FAIL} {name}")
print("=" * 70)
