# Action Required: Send to MerLin Team

## Summary

- **Most important:** `ComputationSpace.UNBUNCHED` returns probabilities renormalised on
  the collision-free subspace. This amounts to postselection, the API does not say so, and
  it silently distorts any shot-budget or detection-efficiency reasoning. We had to switch
  to `FOCK` with threshold detectors.
- **Smaller items:**
  - there is no direct API for mode-pair (k-local) click marginals;
  - the package version string disagrees with the installed distribution;
  - the numpy ≥ 2 requirement breaks shared environments.

## Context

- **Paper:** we reproduced arXiv:2603.13005, a quantum extreme learning machine
  (QELM) on IBM superconducting hardware.
- **Photonic adaptation:** we built a MerLin version of its reservoir.
  - Circuit: a fixed brickwall of beam splitters and phase shifters on 16–24 modes with
    4–6 photons, built as a Perceval circuit inside `merlin.QuantumLayer(experiment=...)`.
  - Readout: click probabilities of mode pairs, estimated with 1e2–1e5 shots.
  - Simulation: analytic, lossless, ideal threshold detectors.
- **Scope:** the tasks were Landsat classification, NARMA forecasting, and output
  variability (concentration).
- **Why it matters:** the paper's claims are about finite-shot readout: how many shots
  are needed, and which features survive shot noise. How MerLin normalises output
  probabilities therefore changes the scientific result, not just the plumbing.
- **Environment:** MerLin 0.4.1 (as reported by `merlin.__version__`), Perceval 1.3.1,
  torch 2.12.1, Python 3.12, macOS CPU.

## Items to Send

### High — ComputationSpace / MeasurementStrategy: `UNBUNCHED` silently renormalises on the collision-free subspace

- **Type:** missing documentation (behaviour is probably intended).
- **What happened:**
  - With `MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED)`, the
    returned distribution sums to 1 over collision-free outcomes only.
  - In effect, every bunched event is discarded and the rest are rescaled. This is
    postselection, and neither the probabilities nor the layer report the discarded mass.
- **Evidence:**
  - Probe on 2026-10-28, 8 modes and 4 photons, using our reservoir circuit. Run under
    `FOCK`, the collision-free outcomes carry 0.202 of the probability. The
    `UNBUNCHED` / `FOCK` ratio for every collision-free outcome was exactly 1 / 0.202.
  - Recorded in `papers/qelm_concentration/LOG.md`, "Phase 4 Adaptation Plan", detector
    and space choice.
- **Why it matters:**
  - `UNBUNCHED` is easily read as "threshold detection". Our internal MerLin cookbook
    annotates it `# threshold detection`, and our reproduction workflow uses it as the
    default. A user who samples shots from it overstates the useful event rate: here by
    about 5x, since about 80 % of events are discarded.
  - The postselection cost drops out of shot-budget, latency and hardware-feasibility
    comparisons.
  - Readout comparisons that depend on shot noise (signal-to-noise eigentasks, NSR
    cutoffs) are biased.
  - At 16–24 modes with 4–6 photons, bunching is not negligible.
- **Suggested action:**
  - State in the `ComputationSpace` and `MeasurementStrategy` docs that `UNBUNCHED`
    conditions on no collisions and renormalises.
  - Expose the kept probability mass (the acceptance rate), for example as an optional
    output or attribute.
  - Point users who need physical click statistics to `FOCK` plus
    `pcvl.Detector.threshold()`. That gives real threshold patterns: 1..n clicks summing
    to 1, with no postselection.

### Medium — QuantumLayer outputs: no direct k-local (mode-pair) click marginals

- **Type:** improvement.
- **What happened:**
  - Local readouts need the 4-outcome click marginal of each mode pair.
  - MerLin returns only the full threshold-pattern distribution. We compute the marginals
    by contracting that distribution with click-indicator matrices built from
    `layer.output_keys`. See `pair_click_probabilities` in
    `papers/qelm_concentration/lib/qelm_merlin.py`.
- **Evidence:**
  - Wall-clock is 1 min 22 s for Landsat at 16 modes / 4 photons, against 44 min 30 s at
    24 modes / 6 photons (`LOG.md`, "Phase 4 MerLin results").
  - We did not profile the run, so we have not confirmed that the full distribution
    dominates this cost. It grows combinatorially with modes and photons.
- **Why it matters:**
  - Local-observable readouts are the standard way to escape concentration in reservoir
    and extreme-learning models, and they need only O(m) marginals.
  - A marginal-only path, or documentation of the contraction pattern, would let users
    scale such models further.
- **Suggested action:**
  - Add a measurement option, or a helper, that returns marginals over given mode subsets
    under threshold or PNR detection.
  - At least add a cookbook example of computing local click marginals from
    `output_keys`.

### Low — Packaging: `merlin.__version__` disagrees with the installed distribution

- **Type:** bug.
- **What happened:** pip reports `merlinquantum 0.4.2` installed, but `merlin.__version__`
  returns `0.4.1`.
- **Evidence:**
  - Fresh conda env, Python 3.12, created 2026-10-23 with
    `pip install ... merlinquantum ...`.
  - Recorded in `papers/qelm_concentration/LOG.md`, "Dependency Additions".
- **Why it matters:**
  - Reproductions record the framework version for provenance, and two different numbers
    make that record ambiguous.
  - We report "0.4.1 as reported by `merlin.__version__`".
- **Suggested action:**
  - Derive `__version__` from package metadata (`importlib.metadata.version`) or the
    build backend.

### Low — Dependencies: numpy ≥ 2 requirement breaks shared scientific environments

- **Type:** pain point.
- **What happened:**
  - Running `pip install merlinquantum` in an existing conda base env upgraded numpy to
    2.x and Perceval to 1.3.1.
  - That broke tensorflow, numba, pennylane and pandas in that env. The base env had to be
    restored by hand, and a dedicated env was created.
- **Evidence:**
  - `papers/qelm_concentration/LOG.md`, "Dependency Additions": the restore commands and
    `pip check` output.
- **Why it matters:**
  - QML users often have PennyLane, TensorFlow or numba in the same environment for
    baselines.
  - The failure appears in other packages, not in MerLin, so the cause is not obvious.
- **Suggested action:**
  - State the numpy ≥ 2 and Perceval version requirements prominently in the install docs,
    and recommend a dedicated environment.
  - If feasible, document which numpy 1.x versions are known to fail.
