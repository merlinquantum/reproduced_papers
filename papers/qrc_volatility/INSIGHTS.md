# INSIGHTS.md — qrc_volatility (arXiv:2505.13933)

Durable observations worth carrying to other reproductions.

## 1. An off-by-one in a baseline's rolling loop can manufacture a quantum advantage

The paper's HAR and HARX losses are inflated because the released notebook reads
the regressor row one month before the forecast target
(`X_test = dff.iloc[end:end+1]` while `actual = dff.iloc[-245:]`). Correcting it
moves HARX from 0.1448 to 0.1016 MSE, past both quantum reservoirs. Nothing about
the quantum model changed.

**Transferable rule.** For every rolling-origin baseline, verify the index
alignment directly: check that the regressor row used for forecast *j* is the row
whose OLS target is the actual value being scored. A cheap positive control is to
assert that a naive persistence forecast (`RV_hat_t = RV_{t-1}`) is *not* better
than the fitted model; a misaligned model will often lose to persistence.

**Why it was findable.** The defect was visible only because the released code
was read line by line against the paper's protocol description. Reproducing the
paper's *numbers* confirmed the bug rather than the model — the number 0.1476 is
reproducible to four decimals *and* wrong.

## 2. "Best of N random instances" is a hyperparameter search, and must be matched

Reporting the best of 100 reservoir draws is only fair if every competitor gets
100 draws under the same rule. Here the paper's headline QRC values sit at roughly
the **5th percentile** of their own 100-instance distributions (QR1 published
0.1050 vs 5th percentile 0.1051; QR2 0.1030 vs 0.1037), while the *typical* draw
(0.1095) is worse than the paper's own best classical baseline.

**Transferable rule.** Whenever a paper reports "the best of N" anything, run all
N, report mean +/- SD and the percentile the published value occupies, and give
the classical control the identical budget and selection rule. Also report a
leakage-free selection variant: here, selecting the instance on a validation
window inside the training sample instead of on the test window costs the QRC
0.0012-0.0035 MSE, which is the whole margin under dispute.

## 3. Published artefacts are the strongest available correctness test

The authors ship `predict_result.csv` (their 245 out-of-sample forecasts) and
`coeff_10.jld2` (their 100 coupling matrices). Reproducing those forecasts to
`atol = 1e-3` — float32 agreement, since the reference is `ComplexF32` — turned a
"does my density-matrix code look right?" question into a hard pass/fail test. It
immediately caught a subtle bug: recovering hidden-state eigenvectors from an SVD
as the rows of `W^dagger` instead of the columns of `W` propagates the complex
*conjugate* hidden state, which changed QR1/QR2 MSE from 0.1051/0.1038 to
0.1112/0.1157 — a plausible-looking wrong answer that no internal consistency
check would have flagged.

**Transferable rule.** Before implementing, inventory the upstream repository for
saved predictions, saved random parameters or saved intermediate tensors, and
convert each into a regression test. Prefer pinning against artefacts over
pinning against printed table values, which are rounded (the paper prints QR2's
0.10375 as "0.103", truncating rather than rounding).

## 4. Exploit state structure before optimising the simulator

The reservoir state is always `rho_hidden (x) |psi_input><psi_input|` with a pure
input factor, so its rank never exceeds `2 ** n_hidden`. Propagating at most 8
state vectors instead of a 1024x1024 density matrix is exact and turned a full
816-month, 10-qubit pass from tens of minutes into ~5 s — which is what made a
200-run instance sweep affordable inside the compute budget.

**Transferable rule.** In "encode a fresh register, evolve, trace out the fresh
register" architectures (quantum reservoirs, QRNN cells, repeated-measurement
schemes) the joint state's rank is bounded by the *retained* subsystem, not by the
total Hilbert space. Look for that bound before reaching for GPUs or tensor
networks.

## 5. The nonlinearity of the feature map was not what helped

Three independent pieces of evidence point the same way:

- A plain rolling ridge on the 21 raw lagged features (`Linear-lag`, 22
  parameters) scores 0.1031, better than both quantum reservoirs.
- In the photonic adaptation, mean test MSE improves monotonically as the encoding
  phase scale is reduced from `pi` to `pi/8`, i.e. as the feature map is driven
  toward its linear limit (PQR1 0.2070 -> 0.1482; PQR2 0.1765 -> 0.1136), at every
  mesh seed and for both variants.
- Widening the photonic readout from 10 to 20 features helps far more
  (0.1715 -> 0.1372) than any change to the quantum dynamics.

**Transferable rule.** For reservoir-computing papers, always include a linear
readout on the raw lagged inputs at matched dimension. If the reservoir does not
beat it, the paper's contribution is a *feature-dimension* effect, not a quantum
one. On strongly persistent targets (log realized volatility, most macro series)
this control is often hard to beat, and a nonlinear feature map can actively hurt.

## 6. Phase encoding is not angle encoding: the periodicity differs

`RY(pi x)|0>` gives amplitudes `cos(pi x / 2), sin(pi x / 2)`, which is injective
for `x` in `[-1, 1]`. A linear-optical phase shifter contributes `exp(i pi x)`,
which is `2 pi`-periodic, so the same `scale = pi` maps `x = -1` and `x = +1` to
the same phase. Transplanting a gate-model encoding scale into a photonic circuit
therefore silently destroys half the feature range. Reducing the scale recovered
0.066 MSE in this reproduction — larger than the entire effect the paper claims.

**Transferable rule.** When porting a gate-model encoding to MerLin, re-derive
the scale from the feature range and the *photonic* periodicity rather than
copying the qubit rotation angle; treat the scale as a predeclared swept
hyperparameter selected on validation data.

## 7. Reproducing a table is not reproducing a claim

Every quantum number in this paper reproduces to three or four decimals, and the
central claim is still unsupported. The three checks that separated the two were:
(a) read the baselines' code, not just their numbers; (b) match the selection
budget; (c) add one iso-dimensional control the paper did not run. None of the
three required more compute than the original experiment.
