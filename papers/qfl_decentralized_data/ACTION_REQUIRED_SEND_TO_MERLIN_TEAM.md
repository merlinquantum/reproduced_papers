# Action Required: Send to MerLin Team

## Summary

Building a federated photonic classifier with MerLin 0.4.1 surfaced five items:

1. A confirmed silent bug: `add_angle_encoding` ignores `scale` when the stage name ends in a digit.
2. A packaging gap that let `merlinquantum 0.3.1` install next to an incompatible Perceval.
3. A constant float buffer in `QuantumLayer.state_dict()` that breaks naive federated averaging.
4. No way to read the post-selection acceptance probability from `mode_expectations(UNBUNCHED)`.
5. Undocumented ordering of input columns across several encoding stages.

## Context

The reproduction concerns Huang, Tan and Xu, "Quantum Federated Learning With Decentralized Data" (IEEE JSTQE 2022). Clients train a variational quantum classifier on private MNIST shards, and a server averages the parameters with FedAvg. As a photonic adaptation, a MerLin classifier replaced the gate circuit: 8 modes, 2 photons, a fixed `add_superpositions` layer, two `add_angle_encoding` stages around trainable `add_entangling_layer` meshes, `ComputationSpace.UNBUNCHED`, and `mode_expectations` into a linear head. It was trained in the same FedAvg loop over 146 runs, all on MerLin 0.4.1 and Perceval 1.3.1 with Python 3.11 and torch 2.12.1. The photonic model worked well: it preserved the paper's behaviour and beat the gate-model reproduction. The items below are what made it harder than necessary or could silently corrupt results for other users. Implementation: `papers/qfl_decentralized_data/lib/vqtn_merlin.py`; tests: `papers/qfl_decentralized_data/tests/test_merlin.py`.

## Items to Send

### High — CircuitBuilder.add_angle_encoding: a stage name ending in a digit silently ignores `scale`

- Type: bug
- What happened: when the encoding stage is named with a trailing digit (for example `name="x0"`), the `scale` argument has no effect. The circuit behaves as if `scale` were 1, with no warning. Letter-only names (for example `"xa"`) respect `scale`.
- Evidence (MerLin 0.4.1, Python 3.11, reproduced 1/1 here and first reported in `papers/vertical_qfl` on 0.4.x):
  ```python
  b = ml.CircuitBuilder(n_modes=4)
  b.add_superpositions(name="sup")
  b.add_angle_encoding(modes=[0, 1, 2, 3], name=NAME, scale=SCALE)
  b.add_superpositions(name="mix")
  layer = ml.QuantumLayer(
      input_size=4,
      builder=b,
      input_state=[1, 0, 1, 0],
      n_photons=2,
      measurement_strategy=ml.MeasurementStrategy.probs(ml.ComputationSpace.UNBUNCHED),
  )
  layer(torch.tensor([[0.1, 0.7, 0.3, 0.5]]))
  ```
  With `NAME="xa"`, scales 1.0 and 3.0 give different outputs (`torch.allclose` False). With `NAME="x0"`, they give identical outputs (`torch.allclose` True).
- Why it matters: the encoding scale is one of the most important hyperparameters. In this reproduction the same circuit scored 96.91 % validation accuracy at scale 1.0 and 87.19 % at scale 0.25, a 9.7-point difference (`results/sweeps/calibration_merlin_e1_0v9.md`). A user who names stages `x0`, `x1`, … gets a silently mis-scaled model and wrong calibration conclusions.
- Suggested action: fix the name parsing (probably a prefix or suffix match on parameter names), or reject such names with an error. Add a test that `scale` changes the output for names with and without trailing digits.

### Medium — Packaging: `merlinquantum 0.3.1` installs next to `perceval-quandela 1.3.1` but fails at import

- Type: bug (probable packaging metadata gap)
- What happened: a fresh venv built with `uv pip install -r requirements.txt`, where the requirements had an unpinned `merlinquantum`, resolved to `merlinquantum==0.3.1` with `perceval-quandela==1.3.1`. Importing MerLin then failed with `ModuleNotFoundError: No module named 'perceval.runtime.session'` (from `merlin/core/merlin_processor.py`, line 17). Pinning `merlinquantum>=0.4` resolved to 0.4.2 and worked. An earlier build with Python 3.14.8 ended with MerLin not importable at all; I did not investigate the cause.
- Evidence: remote venv on an Ubuntu GPU host, Python 3.11.17; the error above was printed by the launch-time import check, and the package lists are recorded in the reproduction's `outdir/remote/env/` freeze files. Uncertainty: I don't know why the resolver preferred 0.3.1 over 0.4.2.
- Why it matters: anyone writing `merlinquantum` without a version bound can get a broken environment that only fails at import, often on a remote machine.
- Suggested action: give older releases an upper bound on `perceval-quandela` (for example `perceval-quandela<1.3` for 0.3.x, or whatever is correct), and declare supported Python versions explicitly so resolvers do not pick unusable combinations.

### Medium — QuantumLayer: constant float buffer in `state_dict()` breaks naive federated averaging

- Type: pain point / missing documentation
- What happened: with `MeasurementStrategy.mode_expectations(...)`, `QuantumLayer.state_dict()` contains a float buffer `measurement_mapping.mask` (shape `(n_modes, n_outcomes)`, e.g. `(8, 70)`) next to the trainable phases. With `probs(...)` it does not. FedAvg implementations commonly average every `state_dict` entry with client weights. For a constant buffer that is mathematically a no-op but numerically not: weights such as 7/31, 11/31, 13/31 do not sum to exactly 1 in floating point, so the mask drifts slightly every round.
- Evidence: `state_dict` keys observed: `{'el': (168,), 'measurement_mapping.mask': (8, 70)}` for `mode_expectations` against `{'el': (168,)}` for `probs`. The reproduction's FedAvg now averages trainable parameters only, and `tests/test_federated.py::test_fedavg_averages_parameters_only_and_keeps_buffers_exactly` checks that buffers stay exactly unchanged.
- Why it matters: federated and distributed training, and model averaging in general (SWA, ensembling by weight averaging), are natural uses of MerLin layers. A silently drifting measurement mask is hard to notice.
- Suggested action: register derived constant tensors with `persistent=False` (or recompute them on load), or document which `state_dict` entries are trainable state and which are constant buffers.

### Medium — MeasurementStrategy.mode_expectations(UNBUNCHED): no access to the post-selection acceptance probability

- Type: improvement (hardware-model concern)
- What happened: in `UNBUNCHED` space, `mode_expectations` returns per-mode occupations of the renormalised collision-free distribution (they sum exactly to the photon number). Nothing is returned about how much probability was discarded by post-selection.
- Evidence: outputs summed to 2.0 and 4.0 for 2 and 4 photons respectively in 8 modes (`tests/test_merlin.py::test_unbunched_threshold_space_is_used`). The reproduction's hardware-aware report has to list the acceptance probability as "not recorded". Uncertainty: I used only the high-level API and did not inspect MerLin's internals for a hidden accessor.
- Why it matters: post-selection cost converts directly into required shots and runtime on hardware. Without it, simulated `UNBUNCHED` accuracies cannot be compared honestly with threshold-detector hardware runs.
- Suggested action: expose the acceptance probability, for example as an optional second output, an attribute after `forward`, or a `return_object=True` field, for `probs` and `mode_expectations` in `UNBUNCHED` and `DUAL_RAIL`.

### Low — QuantumLayer / CircuitBuilder: ordering of input columns across several encoding stages is undocumented

- Type: missing documentation
- What happened: with two `add_angle_encoding` stages (re-uploading), `QuantumLayer` maps the first `k` input columns to the first stage and the next `k` to the second, following builder order. This is not stated in the docstrings or the cookbook, so it had to be checked empirically.
- Evidence: `tests/test_merlin.py::test_input_columns_feed_the_encoding_stages_in_order`. With the first stage's scale set to 0, changing column 1 leaves the output unchanged while changing column 5 alters it. Note that an equal phase shift on every mode is a global phase and invisible, so such tests must shift a single column.
- Why it matters: data re-uploading is a recommended MerLin pattern, and a wrong assumption about column order silently scrambles which features enter which stage.
- Suggested action: document the rule in `QuantumLayer` and `add_angle_encoding`, ideally with `layer.input_parameters` showing the mapping, and add a re-uploading example that shows the column split.
