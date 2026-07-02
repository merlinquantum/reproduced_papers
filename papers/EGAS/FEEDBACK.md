# Feedback on the Reproduction Workflow

## What worked well
- The shared runtime (`implementation.py` + `cli.json` + `lib/runner.py`) made it easy to expose
  multiple experiment "tasks" (`wasserstein`, `fig1`, `egas_eval`, `photonic_eval`) behind one
  entry point without bespoke argparse.
- The "minimal runnable first" doctrine paid off: building a custom batched torch statevector
  engine and validating it to machine precision against PennyLane *before* the search loop
  avoided debugging quantum correctness inside a slow training loop.
- MerLin 0.4.0 shipping a ready `FidelityKernel` removed the need to hand-roll photonic kernels.

## Friction points and missing guidance
- **Parallel-run output collision**: launching several runs in the same second made the shared
  runtime mint identical `run_YYYYMMDD-HHMMSS` directories, so the parallel jobs silently
  overwrote each other's `metrics.json`. Guidance (or a runtime safeguard, e.g. a PID/counter
  suffix) would prevent this. Workaround used: distinct `--outdir` per job.
- The paper-reproduction guide's effort ceilings were useful but the EGAS GPT sampling cost
  (28 sequential transformer forwards per candidate) is a non-obvious bottleneck; a note that
  "autoregressive token generators are dominated by per-step dispatch, not compute" would help.

## Dataset and environment issues
- `ucimlrepo` worked without credentials. Caveat: Wine "color" label lives in a role-`Other`
  column accessible only via `repo.data.original`, not `repo.data.targets`; and some sets expose
  multiple target columns (EGSSD has continuous `stab` + categorical `stabf`) — picking the wrong
  one silently breaks the binary task. Worth a shared helper.
- pennylane / ucimlrepo / POT were not preinstalled; documented installs in `LOG.md`.

## Reproduction ambiguity notes
- The paper underspecifies: GPT architecture/size, the inverse-temperature `γ` in Eq. 10, the
  number of candidates `M` per iteration, the exact two-qubit gate wiring, and the per-dataset
  two-class definitions. All resolved with documented defaults.
- Table I preprocessing is underspecified ("rescaled to [0,2π]"). Per-feature MinMax matches 5/7
  datasets but compresses the two most-separable sets (DB, WC). The absolute W1 magnitudes are
  preprocessing-sensitive; the saturation *ordering* is robust.

## MerLin integration feedback
- See `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`. Headline: `FidelityKernel` is excellent for
  forward kernel evaluation, but *training* a feature-map mesh through it via autograd is very
  slow (~10s+/epoch for 2–3 photons in 8 modes), which limited the trainable-photonic scope.

## Suggestions for improving the guide
- Add a "parallel run hygiene" note (distinct outdir or seed-suffixed run dirs).
- Add a short pattern for GPT/autoregressive generators in QML architecture-search reproductions
  (cost model + the recommendation to reduce iterations rather than model size when CPU-bound).

## Net assessment
The workflow supported a faithful, clearly-labelled reduced reproduction end-to-end. The main
avoidable cost was the parallel-output collision (lost one batch of detailed metrics) and the
photonic-training cost discovery. Both are now documented for future runs.
