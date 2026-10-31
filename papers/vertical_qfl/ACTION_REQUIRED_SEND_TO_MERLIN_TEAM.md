# Action Required: Send to MerLin Team

## Summary

Four MerLin observations from building a photonic vertical-federated-learning model. Each was verified on `merlinquantum` 0.4.2 with Perceval 1.3.1 and torch 2.12.1. A minimal reproduction is included for each.

1. **High, `CircuitBuilder`.** `add_angle_encoding(name=...)` **silently ignores `scale`** when the name ends in a digit, and gives no warning. The same trailing-digit naming merges mesh parameters into a single tensor.
2. **High, `ComputationSpace.UNBUNCHED`.** It is **post-selection on collision-free outcomes followed by renormalisation**. Users may read it as threshold detection: the guide used in our reproduction workflow says so. Nothing exposes the discarded probability. With a single collision-free pattern, the output is constant for every input.
3. **Medium, `MeasurementStrategy.mode_expectations` in `FOCK`.** The output depends only on single-photon interference. It is **exactly constant for one photon per mode**, so a model built on it cannot learn. A built-in threshold (click-probability) readout would help.
4. **Low, packaging.** The `merlinquantum` 0.4.2 distribution reports `merlin.__version__ == "0.4.1"`.

## Context

- **The reproduction:** *Vertical Quantum Federated Learning* (Ballester et al., ICEA '25), in `papers/vertical_qfl`. Clients hold disjoint pixel subsets of the Semeion digits. Each client sends a short message to a server, which classifies 10 classes.
- **The MerLin adaptation (Phase 4):** one 8-mode chip per party (client or server), mostly with 4 photons. Each chip alternates `add_entangling_layer` with `add_angle_encoding` stages (data re-uploading). Clients send mode expectations or click probabilities; the server reads grouped or marginal probabilities.
- **Impact:** items 1-3 each cost a debugging detour.
  - Item 1 made models train identically for different encoding scales, which invalidated a hyperparameter sweep until it was found.
  - Item 2 made our hardware report wrong ("postselection: none"), and it had to be corrected.
  - Item 3 made a whole 8-photon diagnostic train at exactly chance level.
- **Reach:** all of these are silent. No error or warning appears, so other users and automated reproductions are likely to hit them too.

## Items to Send

### High — CircuitBuilder: trailing digits in names silently drop `scale` and merge parameters

- **Type:** bug (probable; the behaviour is confirmed, the root cause is inferred).
- **What happened:**
  - **Scale ignored.** With several encodings in one builder, naming them `x0`, `x1` (or `x0_0`) makes `add_angle_encoding(..., scale=s)` have **no effect**. The outputs are identical for `scale=pi` and `scale=pi/2`. Names without a trailing digit (`xa`, `xb`, `a`, `x`) apply the scale correctly, and then `scale=pi` exactly equals hand-scaling the input by π.
  - **Parameters merged.** Mesh names `mesh0` and `mesh1` are merged into one parameter tensor named `mesh`. It holds 4 × 56 = 224 entries for four 8-mode meshes. The entries stay independent, but the per-stage structure disappears from `named_parameters()`.
  - **Likely cause.** Both symptoms suggest that trailing digits are parsed as an index of the name prefix.
  - **No warning** is emitted in either case.
- **Evidence:**
  - Reproduced on `merlinquantum` 0.4.2 in a fresh process. The script output was `digit names x0,x1: output changes with scale: False` and `letter names xa,xb: output changes with scale: True`. For the mesh names it printed `{'mesh': (24,)}`.
  - The project test `papers/vertical_qfl/tests/test_model_merlin.py::test_encoding_scale_is_applied_with_letter_names` pins the working pattern.
  - `papers/vertical_qfl/LOG.md` ("MerLin API Notes") records the original finding.
  ```python
  import math, torch, merlin as ml
  def layer(names, scale):
      torch.manual_seed(0)
      b = ml.CircuitBuilder(n_modes=4)
      for n in names:
          b.add_entangling_layer(name="m" + n[1:].replace("0", "a").replace("1", "b"))
          b.add_angle_encoding(modes=[0, 1, 2, 3], name=n, scale=scale)
      return ml.QuantumLayer(input_size=4 * len(names), builder=b, input_state=[1, 0, 1, 0], n_photons=2,
          measurement_strategy=ml.MeasurementStrategy.mode_expectations(ml.ComputationSpace.UNBUNCHED))
  x = (torch.rand(3, 8) > 0.5).float()
  print(torch.allclose(layer(["x0", "x1"], math.pi)(x), layer(["x0", "x1"], math.pi / 2)(x)))  # True: scale ignored
  print(torch.allclose(layer(["xa", "xb"], math.pi)(x), layer(["xa", "xb"], math.pi / 2)(x)))  # False: scale applied
  ```
- **Why it matters:**
  - Numbered names (`enc0`, `enc1`, ...) are the natural way to build data re-uploading circuits in a loop, which the cookbook recommends.
  - The failure is silent. In our case a scale sweep returned identical results for different scales, and the cookbook names the encoding scale as a decisive hyperparameter.
- **Suggested action:**
  - Either apply `scale` per encoding regardless of the name format, or raise an error or warning when a name collides with the indexed-prefix convention.
  - Document the naming rules for multiple encodings and meshes.

### High — ComputationSpace.UNBUNCHED: document it as post-selection and expose the discarded mass

- **Type:** missing documentation / API confusion. The behaviour itself is physically consistent once it is understood as post-selection.
- **What happened:**
  - **It is post-selection.** `UNBUNCHED` evolves the full Fock state exactly. It then keeps only the collision-free output patterns and **renormalises** them. It does not model threshold detectors, which would merge bunched events into click patterns rather than discard them.
  - **It is silent.** Nothing reports the discarded probability, which is the success probability that a hardware run would pay.
  - **It can make a model constant.** When the input state allows only one collision-free pattern, the output is 1.0 for every input. With a Hong-Ou-Mandel input, all input dependence vanishes without a warning.
- **Evidence:**
  - **Hong-Ou-Mandel test.** Input |1,1⟩ through BS – PS(x) on mode 0 – BS, for x = 0, 0.5, 1, 1.5:
    - `FOCK` returns P(1,1) = 1.0, 0.77, 0.292, 0.005, equal to cos²(x).
    - `UNBUNCHED` returns a single key `(1, 1)` with probability `1.0` for every x.
  - **Three modes, input |1,1,0⟩.** The `UNBUNCHED` probabilities equal the `FOCK` collision-free probabilities divided by their sum, to 1e-5. Between 35% and 71% of the probability was discarded, depending on x.
  - **Sources:** `papers/vertical_qfl/LOG.md` ("Correction — UNBUNCHED semantics") and the reproduction snippet below.
  - **Documentation:** the reproduction guide we used (an internal MerLin cookbook, section 7) equates `UNBUNCHED` with threshold detectors. We did not check whether the official MerLin user guide says the same.
  ```python
  import torch, perceval as pcvl, merlin as ml
  c = pcvl.Circuit(2); c.add(0, pcvl.BS()); c.add(0, pcvl.PS(pcvl.P("px1"))); c.add(0, pcvl.BS())
  for space in (ml.ComputationSpace.UNBUNCHED, ml.ComputationSpace.FOCK):
      q = ml.QuantumLayer(input_size=1, circuit=c, input_parameters=["px"], trainable_parameters=[], input_state=[1, 1],
          n_photons=2, measurement_strategy=ml.MeasurementStrategy.probs(computation_space=space))
      print(space, list(q.output_keys), q(torch.tensor([[0.0], [1.0]])))
  ```
- **Why it matters:**
  - **Hardware reasoning.** Results computed in `UNBUNCHED` describe a post-selected experiment. Their sampling cost includes the discarded fraction. For 4 photons in 8 modes under uniform mixing, only about 70/330 ≈ 21% of events are collision-free.
  - **Expressivity.** The discarded bunched events carry part of the multi-photon signal that drives photon-number expressivity results, as in Gan et al. (`papers/fock_state_expressivity`).
  - **Reporting.** Users who believe they model threshold detectors will report "no post-selection", as we initially did.
- **Suggested action:**
  - State explicitly in the docs and docstrings that `UNBUNCHED` is post-selection with renormalisation.
  - Expose the kept probability mass per sample, for example as an optional output or attribute, so users can report the success probability.
  - Warn when the collision-free space has a single outcome.
  - Point users who want threshold detectors to the proper detector model (see the next item).

### Medium — MeasurementStrategy.mode_expectations (FOCK): blind to multi-photon interference; a built-in click-probability readout would help

- **Type:** missing documentation / improvement. The physics is expected, but the pitfall is not obvious.
- **What happened:**
  - **Why it is blind.** In `FOCK`, the mean photon number per mode is ⟨n_k⟩ = Σ_j |U_kj|² n_j. It depends only on single-photon (classical-wave) interference.
  - **Why the 8-photon model failed.** With one photon per mode, every ⟨n_k⟩ is exactly 1 for any unitary, by unitarity. A model whose client sends `mode_expectations` with input `[1]*8` therefore receives a constant message. It trained at exactly chance level, with client gradients of about 4e-8.
  - **What fixed it.** Replacing the message by per-mode click probabilities P(n_k ≥ 1), computed by summing `probs` over outcomes, made the same 8-photon chips learn.
- **Evidence:**
  - Reproduced: a 4-mode chip with input |1,1,1,1⟩ and `mode_expectations(FOCK)` returns `[1.0, 1.0, 1.0, 1.0]` for 3 random inputs.
  - `papers/vertical_qfl/LOG.md` ("D2 result" and "D3 result") records the chance-level runs and the fix.
  - The test `tests/test_model_merlin.py::test_mean_photon_number_message_is_blind_with_one_photon_per_mode` reproduces it.
- **Why it matters:** `mode_expectations` is the natural fixed-size readout for passing information between photonic layers or parties. Users who add photons to gain expressivity will see none of the gain through this readout, and get a constant output in the one-photon-per-mode case.
- **Suggested action:**
  - Document this property of `mode_expectations` next to the photon-number expressivity guidance.
  - Consider a built-in per-mode click-probability measurement (threshold detection without post-selection) as a `MeasurementStrategy`, so users do not have to compute marginals from `output_keys` by hand.
  - The layer exposes `detectors` and `has_custom_detectors`; if threshold detectors are already supported through custom detectors, an example would cover this. We did not test that path.

### Low — Packaging: `merlinquantum` 0.4.2 reports `merlin.__version__ == "0.4.1"`

- **Type:** bug (packaging metadata).
- **What happened:** `importlib.metadata.version("merlinquantum")` returns `0.4.2`, but `merlin.__version__` returns `0.4.1`.
- **Evidence:** the reproduction script printed `merlinquantum dist 0.4.2 | merlin.__version__ 0.4.1`. Every run log of the shared runtime prints `MerLin version 0.4.1` from `merlin.__version__`. The project's phase audit flagged the resulting inconsistency (`papers/vertical_qfl/LOG.md`, "Phase 4 MerLin Audit").
- **Why it matters:** reproducibility records cite the version from `__version__`. A mismatch with the installed distribution makes environments ambiguous.
- **Suggested action:** bump `merlin.__version__`, or derive it from the package metadata.

### Low — CircuitBuilder.add_superpositions: `trainable_phi=False` is ignored

- **Type:** bug or missing documentation.
- **What happened:** `add_superpositions(trainable_theta=True, trainable_phi=False)` on 6 modes gives the same 10 trainable parameters (2 per beam splitter) as `trainable=True`; with `depth=2` it gives 20.
- **Evidence:** probe on merlinquantum 0.4.2 (`QuantumLayer.named_parameters()` of a 6-mode builder with one superposition layer and an angle encoding). Seen while building the D5 control in `papers/vertical_qfl`.
- **Why it matters:** an angle-only mixer would halve the parameter count, and users cannot build it from the documented flags.
- **Suggested action:** honour the flag or document that the phase is always trainable.
