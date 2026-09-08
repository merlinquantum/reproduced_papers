"""Preflight anchors that must pass on the GPU host before a production queue.

Two of these cannot run anywhere else: the GPU Clifford & Clifford sampler has to
agree with Perceval's on the same unitary, and the batched multi-circuit call has
to agree with one call per circuit. If either is wrong every number the queue
produces is wrong in a way that looks plausible.

    python scripts/gpu_anchors.py [--device cuda]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bbs_gpu import solve_batch                                           # noqa: E402
from lib.problems_torch import build_batch                                    # noqa: E402
from lib.sampler_torch import draw_clicks                                     # noqa: E402
from lib.tbi import alternating_input, beamsplitter_layout, unitary           # noqa: E402

try:  # Perceval is only needed by the cross-check below, not by the queue itself
    from lib.sampler import ClickSource
except ImportError:  # pragma: no cover - environment dependent
    ClickSource = None


def check_boson_matches_perceval(device, m=12, delays=(1, 3, 9), shots=40000):
    """GPU boson sampler vs Perceval's, same unitary, per-mode click rates."""
    if ClickSource is None:
        print("  boson vs Perceval: SKIPPED (perceval-quandela not installed on this host)")
        return
    theta = np.random.default_rng(0).random(len(beamsplitter_layout(m, delays))) * 2 * np.pi
    input_state = alternating_input(m)
    modes = [i for i, n in enumerate(input_state) if n]
    reference = ClickSource(m, delays, input_state, source="boson").draw(theta, shots, np.random.default_rng(1))
    unitaries = unitary(torch.tensor(theta, dtype=torch.float64), m, delays)[None].to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    ours = draw_clicks(unitaries, modes, shots, generator, "boson", m).to(torch.float64)
    gap = np.abs(reference.mean(0) - ours.mean(dim=(0, 1)).cpu().numpy()).max()
    tolerance = 4 * math.sqrt(0.25 / shots)
    print(f"  boson vs Perceval: max per-mode rate gap {gap:.4f} (tolerance {tolerance:.4f}) "
          f"| mean clicks {reference.mean(0).sum():.3f} vs {float(ours.mean(dim=(0,1)).sum()):.3f}")
    assert gap < tolerance, "GPU boson sampler disagrees with Perceval"


def check_batched_matches_per_circuit(device, m=10, delays=(1, 3), shots=20000):
    """B distinct unitaries in one call must equal B calls of one unitary each."""
    rng = np.random.default_rng(1)
    angles = rng.random((4, len(beamsplitter_layout(m, delays)))) * 2 * np.pi
    modes = [i for i, n in enumerate(alternating_input(m)) if n]
    stacked = unitary(torch.tensor(angles, dtype=torch.float64), m, delays).to(device)
    together = draw_clicks(stacked, modes, shots, torch.Generator(device=device).manual_seed(2),
                           "boson", m).to(torch.float64).mean(dim=1).cpu().numpy()
    separate = np.stack([
        draw_clicks(stacked[i: i + 1], modes, shots, torch.Generator(device=device).manual_seed(3 + i),
                    "boson", m).to(torch.float64).mean(dim=1)[0].cpu().numpy()
        for i in range(4)
    ])
    gap = np.abs(together - separate).max()
    tolerance = 5 * math.sqrt(0.25 / shots)
    print(f"  batched vs per-circuit: max per-mode rate gap {gap:.4f} (tolerance {tolerance:.4f})")
    assert gap < tolerance, "batched sampler does not match per-circuit sampling"


def check_accounting(device):
    """Every run must evaluate exactly Appendix B's bound."""
    seeds = [0, 1]
    cost_batch, _ = build_batch("knapsack", 8, seeds, device)
    result = solve_batch(cost_batch, 8, seeds, updates=4, samples=4, source="bernoulli", device=device)
    print(f"  accounting: {result['evaluations']} evaluations vs bound {result['budget']}")
    assert result["evaluations"] == result["budget"]


def check_end_to_end(device):
    """A short boson run must train and find small optima."""
    seeds = list(range(4))
    cost_batch, optima = build_batch("knapsack", 10, seeds, device)
    result = solve_batch(cost_batch, 10, seeds, updates=200, samples=50, source="boson", device=device)
    found = sum(abs(float(b) - o) < 1e-9 for b, o in zip(result["best_cost"], optima))
    print(f"  end to end m=10: {found}/4 optimal, E[C] {np.mean(result['history'][:10]):.1f} "
          f"-> {np.mean(result['history'][-10:]):.1f}, clicks {np.mean(result['mean_clicks']):.2f}/10")
    assert found >= 3, "boson path failed to solve size-10 knapsack"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-sampler", action="store_true", help="skip the checks needing cliffordgpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"anchors on {device}")
    check_accounting(device)
    if not args.skip_sampler:
        check_boson_matches_perceval(device)
        check_batched_matches_per_circuit(device)
        check_end_to_end(device)
    print("OK")


if __name__ == "__main__":
    main()
