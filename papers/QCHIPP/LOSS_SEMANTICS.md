# Q-CHIPP source loss semantics

## Finding and provenance

**There is no automatic signed-expectation-to-probability conversion in this source path.** With estimator output shape `(1,)`, numeric labels `{-1,+1}`, and `loss="cross_entropy"`, Qiskit Machine Learning 0.9.0 selects `BinaryObjectiveFunction` and applies its generic `CrossEntropyLoss` directly to the signed scalar. It does not construct a two-class probability vector, apply a sigmoid, or insert a complementary `(1-p)` term.[2][3][4]

This analysis inspected IBM/Q-CHIPP v1.0.0 (commit `9f72bd9fc87522bc45c806f9849d07210a64e48d`) and the public qiskit-machine-learning 0.9.0 tag. The author release pins `qiskit-machine-learning==0.9.0` in `requirements.txt:12`.[1] The four Qiskit files listed under Sources were downloaded from the matching public `0.9.0` tag, all HTTP 200. No Qiskit or model packages were installed.

## Exact forward and dispatch chain

1. `qcnn_all.py:191–204` hard-codes `pad_observable=False`; `:262–283` passes configured loss to `get_model` and then calls `fit`.[1]
2. `libraries/qcnn_lib.py:109–168` sets `one_hot=(primitive == 'sampler')`. Estimator classification therefore uses `one_hot=False`; its SPSA/COBYLA/L_BFGS_B constructors all pass the loss unchanged.[1]
3. `qcnn_lib.py:219–242` constructs one observable. For the even-width pyramid, writing `c=n/2-1`, the Pauli label is `I^c Z I^(n-c-1)` with coefficient `0.5`. Under Qiskit little-endian indexing the measured logical wire is `q=n-c-1=n/2`, not `n/2-1`. For nonpyramid readout, leftmost Z is wire `n-1`; IBM layouts additionally remap this observable.[1]
4. The author's custom `libraries/estimator_qnn.py:169–175,209–237` sets `output_shape=(len(observables),)=(1,)` and reshapes `results.data.evs` into `(N,1)`. It introduces no activation or affine mapping.[1]
5. Qiskit `trainable_model.py:80–89` resolves the string to `CrossEntropyLoss`. `neural_network_classifier.py:118–140` selects `BinaryObjectiveFunction` solely because `output_shape==(1,)`. Numeric labels pass unchanged through `_validate_input` (`:235–246`); binary validation (`:278–292`) checks one-dimensional shape and exactly two unique values, **not** membership in `{-1,+1}`.[2][5]
6. `objective_functions.py:112–117` reshapes target to `(N,1)` and computes the mean of the scalar loss. Its multiclass and one-hot objectives are not reached.[3]

For a normalized ideal output state, define

\[
z_i=\langle\psi(x_i,\theta)|Z_q|\psi(x_i,\theta)\rangle,\qquad s_i=z_i/2\in[-1/2,1/2].
\]

These are derived forward semantics of the inspected observable. Error-mitigated hardware estimates need not obey the ideal interval exactly. The built-in `predict` uses `np.sign(s)` (`neural_network_classifier.py:168–171`), so exact zero produces **0**, not +1. `predict_proba` simply returns the QNN forward values (`:208–211`); its name does not make these signed values probabilities.[1][2]

## Exact scalar objective, not Bernoulli BCE

`loss_functions.py:149–163` evaluates `-einsum(target, log2(clip(predict, 1e-10, None)))`, reducing across the output dimension. Since that dimension is one, the exact objective is[3][4]

\[
\epsilon=10^{-10},\quad C(s)=\max(s,\epsilon),\qquad
\ell(s_i,y_i)=-y_i\log_2 C(s_i),\qquad
J(\theta)=-\frac1N\sum_{i=1}^{N}y_i\log_2 C(s_i(\theta)).
\]

For `y=+1`, this is `-log2 C(s)`; for `y=-1`, it is `+log2 C(s)`. All `s<=epsilon` are a flat plateau. Negative-class loss is negative, reaching approximately `-33.219280948873624`; positive-class loss there is approximately `+33.219280948873624`. At the largest ideal score `s=0.5`, their losses are `-1` and `+1`. These numbers were obtained by executing the downloaded loss definition with NumPy, not guessed or simulated training results.

| s | loss y=-1 | loss y=+1 |
|---:|---:|---:|
| -0.5 | -33.219280948873624 | 33.219280948873624 |
| -0.25 | -33.219280948873624 | 33.219280948873624 |
| 0 | -33.219280948873624 | 33.219280948873624 |
| 0.25 | -2 | 2 |
| 0.5 | -1 | 1 |

Consequences (mathematical deductions from this expression): the negative class is rewarded for crossing into the clipped plateau but receives no further finite-difference preference among negative scores; positive examples in the plateau have no local finite-difference signal until perturbations cross it. The mean may be negative and class contributions can cancel. A downward trace is not proof of a valid probability-based cross-entropy objective.

### Gradient API mismatch

`CrossEntropyLoss.gradient` explicitly says it assumes softmax and returns `predict*sum(target)-target` (`loss_functions.py:165–177`). For the single signed output, the returned value is `g_i=y_i(s_i-1)`. `BinaryObjectiveFunction.gradient` contracts this with the QNN backward derivatives and averages (`objective_functions.py:125–138`):[3][4]

\[
G_{\rm API}(\theta)=\frac1N\sum_i y_i(s_i-1)\nabla_\theta s_i.
\]

This is **not** the derivative of the actual clipped log-base-2 objective. Away from its clipping boundary, the true scalar derivative is

\[
\frac{\partial\ell}{\partial s}=
\begin{cases}
0,&s<\epsilon,\\
-y/(s\ln2),&s>\epsilon.
\end{cases}
\]

For example, executing the source gives gradient-API values `+1.25` (y=-1) and `-1.25` (y=+1) at `s=-0.25`, while the true local derivative is zero. Finite-difference/value-based SPSA and COBYLA use objective values, so matching those values is the key reproduction requirement; a gradient-based route must explicitly choose whether to emulate the source gradient API or differentiate the actual scalar loss. Do not assume the two are equivalent.

## Important label caveat: release defaults are not guaranteed signed

`configs/defaults_qcnn.yaml:23–24,34–35,54–55` sets `Tcell_response`, null `label_binarization_threshold`, estimator/cross_entropy, and `min_label_value=-1`. However, `mylib.py:334–344` remaps labels to `min_label_value` **only when a threshold is non-null**. The released split 1 train/test CSVs were inspected with Python `csv.DictReader`: both contain `Tcell_response` values `{"0","1"}`. Thus the supplied default preprocessing keeps numeric `{0,1}`; setting `min_label_value=-1` alone does not establish signed training targets.[1]

For `{0,1}`, the same dispatch yields `ell(s,0)=0` and `ell(s,1)=-log2 C(s)`: negative examples contribute no objective penalty at all. The classifier nevertheless predicts by sign. `qcnn_all.py:317–320` converts predicted -1 to 0 and everything else (including exact zero) to 1 for its final classification report. This default-path issue must be distinguished from the explicitly requested `{-1,+1}` analysis. Historical experiment settings/serialized targets are needed before asserting that every published run used either label convention.[1][2]

## Faithful reproduction versus a repaired probability loss

**Source-faithful option:** reproduce the exact logical wire, `s=0.5*<Z>`, `(N,1)` shape, actual experiment label values, lower clip `1e-10` without an upper clip, base-2 logarithm, sample mean, and sign prediction. An external statevector circuit evaluator can implement this forward/loss without installing the old Qiskit stack. Test source-equivalent loss at the scores above, clipping-boundary behavior, sign-at-zero policy, and the label preprocessing. Matching forward/loss is possible; it does not validate all published metrics or establish equivalence of simulator noise, SPSA schedules, warm-start state, or hardware.

**Explicitly repaired option:** because `s=z/2`, the physical probability of Z=+1 (qubit outcome zero) is

\[
p_+(x,\theta)=\frac{1+z}{2}=\frac12+s,
\qquad p_-=\frac12-s,
\qquad t=\frac{y+1}{2},
\]

and a well-defined two-class loss is

\[
J_{\rm BCE}=-\frac1N\sum_i\left[t_i\log_2 p_{+,i}+(1-t_i)\log_2 p_{-,i}\right]
\]

with an explicit numerical clipping policy for both probabilities. Note that `(1+s)/2` is **not** the physical probability for an observable already scaled by 0.5. The mathematical class assignment above aligns positive class with positive Z; different biological assignments must be declared.

This repair changes training and must be labeled a deviation, not silently presented as the author/Qiskit objective. Qiskit's sampler branch (`qcnn_lib.py:243–263`) has two outputs and `one_hot=True`, reaching `OneHotObjectiveFunction`, but sampler `last_qubit` reads the leftmost measured bit, not the pyramid central wire; changing primitives is not automatically equivalent.[1][2][3]

Adding the optional `0.5*I` observable offset alone is insufficient: that branch measures the last wire and the classifier still uses the scalar binary objective and sign threshold rather than a two-class Bernoulli loss (`qcnn_lib.py:224–230`).[1][2][3]

## Verification scope

Downloaded definitions were AST-selected and executed unmodified, with only import dependencies stubbed and a deterministic fake QNN supplying known scalar scores/derivatives. This tests the actual loss, objective, and classifier dispatch definitions, **not** a full Qiskit install or a quantum-training run. The probe selected `BinaryObjectiveFunction`; on scores `[-0.25,0.25]` with targets `[-1,+1]` it returned objective `-15.609640474436812` and gradient API `[[0.25]]`. Main paper analysis and implementation were not edited.

## Sources

[1] https://github.com/IBM/Q-CHIPP/tree/9f72bd9fc87522bc45c806f9849d07210a64e48d
[2] https://raw.githubusercontent.com/qiskit-community/qiskit-machine-learning/0.9.0/qiskit_machine_learning/algorithms/classifiers/neural_network_classifier.py
[3] https://raw.githubusercontent.com/qiskit-community/qiskit-machine-learning/0.9.0/qiskit_machine_learning/algorithms/objective_functions.py
[4] https://raw.githubusercontent.com/qiskit-community/qiskit-machine-learning/0.9.0/qiskit_machine_learning/utils/loss_functions/loss_functions.py
[5] https://raw.githubusercontent.com/qiskit-community/qiskit-machine-learning/0.9.0/qiskit_machine_learning/algorithms/trainable_model.py
