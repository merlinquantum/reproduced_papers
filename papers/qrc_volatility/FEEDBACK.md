# FEEDBACK.md — reproduction workflow feedback (qrc_volatility)

## What worked well

1. **"Minimal runnable first" paid for itself immediately.** Reconstructing the
   dataset and reproducing a single HAR baseline before writing any quantum code
   took ~15 minutes and produced the reproduction's most important finding (the
   HAR indexing defect). Had implementation started with the reservoir, that
   check would probably have come late or not at all.
2. **The mandate to hunt for released artefacts** (`find-and-download-data`,
   VISITED_URLS caching) led to `predict_result.csv` and `coeff_10.jld2`, which
   converted correctness from a judgement call into a regression test and caught a
   real bug. This is worth promoting from "nice to have" to an explicit Phase 1
   checklist item: *inventory upstream saved predictions / saved random parameters
   and turn each into a test before implementing*.
3. **Sweep-integrity rules** (predeclare the selection metric, split, direction
   and tie rule before launching) mapped exactly onto the paper's central weakness.
   Because the split had to be declared in advance, the reproduction naturally
   produced both the paper's protocol and a leakage-free one, which is what made
   the fairness finding quantitative rather than rhetorical.
4. **Separating metric agreement from claim support** in the verdict vocabulary
   was essential here. "Quantitatively reproduced, claim unsupported" is the
   accurate summary and the policy gave it a name.

## Friction and suggested improvements

1. **The paper template's `lib/__init__.py` computes the repository root as
   `parents[2]`, which resolves to `papers/` rather than the repository root**, so
   `from runtime_lib import config` fails for any test or script that imports
   `lib.*` without the repo root already on `sys.path`. It needs `parents[3]`.
   This is a template bug affecting every new paper; worth fixing upstream in
   `papers/reproduction_template/lib/__init__.py`.
2. **The template also copies `implementation.py` into each paper folder**, which
   defeats `runtime_lib.data_paths.find_repo_root` (it looks for
   `implementation.py` or `.git` walking up from the cwd). The runtime therefore
   resolved `data_root` to `papers/<paper>/data` instead of `<repo>/data`. Every
   paper has to work around this; either drop the per-paper `implementation.py` or
   make `find_repo_root` require a stronger marker.
3. **The template's `tests/test_smoke.py` calls
   `runner.train_and_evaluate(cfg, run_dir)` with the *full default config*.** For
   a scaffold that writes `done.txt` this is instant; the moment the runner is
   real, `pytest` silently launches the paper-accurate experiment. It hung the
   suite here. The template test should use an explicitly reduced config or mock
   the runner.
4. **Validate the logging contract on a smoke run at the end of Phase 2, and make
   the skill say so louder.** The skill does say "Validate a smoke run with
   `validate_logging.py`", but it reads as an optional closing note rather than a
   phase gate, and nothing in the phase files re-states it. Skipping it here let
   five evidence defects (relative artifact paths, duplicated `DATASET_READY`,
   missing per-candidate `DATASET_READY`, a seed mismatch, a missing
   `CANDIDATE_STARTED`) survive into four expensive runs, discovered only in
   Phase 8 and costing three regeneration passes. Every one would have been caught
   in about a minute by validating the smoke run and the smoke sweep that already
   existed. Suggestion: make "logging validation passes on a smoke run and a smoke
   sweep" an explicit Phase 2 exit condition, and have the validator's own error
   text point at the emitting-code fix rather than only the symptom.

5. **Validate *all* sweep candidates, not a sample, and fix path contracts by
   exhaustive grep.** Two of my three failed attempts came from fixing the sites
   named in an error message instead of auditing every path construction, and from
   validating four sampled candidates instead of all 200. `validate_logging.py`
   accepts repeated `--run-dir`, so full validation is one shell expansion away and
   costs seconds; there is no reason to sample.

6. **The experiment-logging contract does not say how a sweep coordinator should
   write per-candidate `run.log` files.** `configure_logging` uses
   `logging.basicConfig(force=True)`, so calling it per candidate would steal the
   coordinator's handler. The contract says "parallel workers write isolated run
   logs; only the coordinator writes the shared sweep log", but in a *sequential*
   in-process sweep there are no workers. This reproduction writes candidate logs
   directly with small helpers; a documented pattern (or a
   `configure_logging(..., isolated=True)` returning a handle) would remove the
   guesswork.
7. **`arch.bootstrap.MCS` raises `IndexError` rather than degrading on short
   samples** (it cannot eliminate any model, so `loc.squeeze()[0]` indexes an
   empty array). Any reproduction that runs a smoke config through an MCS step
   hits this. Worth a note in the shared guidance: guard MCS on a minimum sample
   length.
8. **Guidance on shell hygiene would have saved time.** Three separate commands
   here were killed because a `pkill -f <pattern>` / `pgrep -f <pattern>` pattern
   also matched the agent's own wrapping shell command line. A one-line warning in
   the execution-governance rules ("never `pkill -f` on a string that appears in
   your own command; match on PID") would be a cheap fix.
9. **Cost governance vs. the multi-seed requirement.** The statistical-interpretation
   rule asks for >= 3 seeds on final tables, but for this paper the scientifically
   necessary replication was 100 reservoir draws (because the paper selects the
   best of 100), while several models are fully deterministic given their draw. The
   policy would be clearer if it framed the requirement as "replicate over the
   sources of randomness the paper exploits, at the budget the paper uses", rather
   than a fixed seed count.

## Non-issues worth recording

- The 30-60 minute dependency-debugging ceiling was never approached: everything
  needed beyond the image was `pip install statsmodels arch`.
- The throughput-oriented stopping rule was applied twice, to drop the LSTM
  baselines and the Shapley analysis. Both drops are recorded as deviations with
  reasons, and neither is close to the decision boundary. That felt like the rule
  working as intended rather than a shortcut.
