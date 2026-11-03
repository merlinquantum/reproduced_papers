# Protocol sensitivity summary

Perspective: `delta_f1 = model F1 - baseline F1`; wins and losses use the same first-named-model perspective.

W/T/L is counted seed by seed with `np.isclose(..., atol=1e-12, rtol=0)`.

| protocol_id | preprocessing_protocol | trace_protocol | kernel_normalization | q | model | baseline | mean_f1 | std_f1 | baseline_mean_f1 | baseline_std_f1 | delta_f1 | wins | ties | losses | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_leak__legacy_trace | legacy_train_plus_heldout |  | none | 4 | merlin_fidelity | linear_c1 | 0.757814 | 0.131587 | 0.758009 | 0.079520 | -0.000195 | 6 | 0 | 4 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout |  | none | 4 | merlin_fidelity | rbf_tuned | 0.757814 | 0.131587 | 0.759963 | 0.121262 | -0.002150 | 4 | 2 | 4 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout |  | none | 6 | merlin_fidelity | linear_c1 | 0.770021 | 0.096427 | 0.786008 | 0.094895 | -0.015987 | 2 | 3 | 5 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout |  | none | 6 | merlin_fidelity | rbf_tuned | 0.770021 | 0.096427 | 0.818913 | 0.090559 | -0.048891 | 1 | 1 | 8 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout | legacy_square_only |  | 4 | qsvm | linear_c1 | 0.808947 | 0.059340 | 0.758009 | 0.079520 | 0.050938 | 7 | 0 | 3 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout | legacy_square_only |  | 4 | qsvm | rbf_tuned | 0.808947 | 0.059340 | 0.759963 | 0.121262 | 0.048984 | 6 | 1 | 3 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout | legacy_square_only |  | 6 | qsvm | linear_c1 | 0.822105 | 0.077912 | 0.786008 | 0.094895 | 0.036097 | 7 | 0 | 3 | 10 |
| legacy_leak__legacy_trace | legacy_train_plus_heldout | legacy_square_only |  | 6 | qsvm | rbf_tuned | 0.822105 | 0.077912 | 0.818913 | 0.090559 | 0.003193 | 4 | 1 | 5 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout |  | train_trace | 4 | merlin_fidelity | linear_c1 | 0.000000 | 0.000000 | 0.758009 | 0.079520 | -0.758009 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout |  | train_trace | 4 | merlin_fidelity | rbf_tuned | 0.000000 | 0.000000 | 0.759963 | 0.121262 | -0.759963 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout |  | train_trace | 6 | merlin_fidelity | linear_c1 | 0.000000 | 0.000000 | 0.786008 | 0.094895 | -0.786008 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout |  | train_trace | 6 | merlin_fidelity | rbf_tuned | 0.000000 | 0.000000 | 0.818913 | 0.090559 | -0.818913 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout | train_trace |  | 4 | qsvm | linear_c1 | 0.000000 | 0.000000 | 0.758009 | 0.079520 | -0.758009 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout | train_trace |  | 4 | qsvm | rbf_tuned | 0.000000 | 0.000000 | 0.759963 | 0.121262 | -0.759963 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout | train_trace |  | 6 | qsvm | linear_c1 | 0.000000 | 0.000000 | 0.786008 | 0.094895 | -0.786008 | 0 | 0 | 10 | 10 |
| legacy_leak__train_trace | legacy_train_plus_heldout | train_trace |  | 6 | qsvm | rbf_tuned | 0.000000 | 0.000000 | 0.818913 | 0.090559 | -0.818913 | 0 | 0 | 10 | 10 |
| train_only__legacy_trace | train_only |  | none | 4 | merlin_fidelity | linear_c1 | 0.755433 | 0.133619 | 0.758009 | 0.079520 | -0.002576 | 6 | 0 | 4 | 10 |
| train_only__legacy_trace | train_only |  | none | 4 | merlin_fidelity | rbf_tuned | 0.755433 | 0.133619 | 0.759963 | 0.121262 | -0.004531 | 4 | 2 | 4 | 10 |
| train_only__legacy_trace | train_only |  | none | 6 | merlin_fidelity | linear_c1 | 0.770021 | 0.096427 | 0.786008 | 0.094895 | -0.015987 | 2 | 3 | 5 | 10 |
| train_only__legacy_trace | train_only |  | none | 6 | merlin_fidelity | rbf_tuned | 0.770021 | 0.096427 | 0.814960 | 0.086978 | -0.044939 | 1 | 1 | 8 | 10 |
| train_only__legacy_trace | train_only | legacy_square_only |  | 4 | qsvm | linear_c1 | 0.813727 | 0.057270 | 0.758009 | 0.079520 | 0.055718 | 8 | 0 | 2 | 10 |
| train_only__legacy_trace | train_only | legacy_square_only |  | 4 | qsvm | rbf_tuned | 0.813727 | 0.057270 | 0.759963 | 0.121262 | 0.053763 | 6 | 1 | 3 | 10 |
| train_only__legacy_trace | train_only | legacy_square_only |  | 6 | qsvm | linear_c1 | 0.822105 | 0.077912 | 0.786008 | 0.094895 | 0.036097 | 7 | 0 | 3 | 10 |
| train_only__legacy_trace | train_only | legacy_square_only |  | 6 | qsvm | rbf_tuned | 0.822105 | 0.077912 | 0.814960 | 0.086978 | 0.007145 | 4 | 1 | 5 | 10 |
| train_only__train_trace | train_only |  | train_trace | 4 | merlin_fidelity | linear_c1 | 0.000000 | 0.000000 | 0.758009 | 0.079520 | -0.758009 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only |  | train_trace | 4 | merlin_fidelity | rbf_tuned | 0.000000 | 0.000000 | 0.759963 | 0.121262 | -0.759963 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only |  | train_trace | 6 | merlin_fidelity | linear_c1 | 0.000000 | 0.000000 | 0.786008 | 0.094895 | -0.786008 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only |  | train_trace | 6 | merlin_fidelity | rbf_tuned | 0.000000 | 0.000000 | 0.814960 | 0.086978 | -0.814960 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only | train_trace |  | 4 | qsvm | linear_c1 | 0.000000 | 0.000000 | 0.758009 | 0.079520 | -0.758009 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only | train_trace |  | 4 | qsvm | rbf_tuned | 0.000000 | 0.000000 | 0.759963 | 0.121262 | -0.759963 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only | train_trace |  | 6 | qsvm | linear_c1 | 0.000000 | 0.000000 | 0.786008 | 0.094895 | -0.786008 | 0 | 0 | 10 | 10 |
| train_only__train_trace | train_only | train_trace |  | 6 | qsvm | rbf_tuned | 0.000000 | 0.000000 | 0.814960 | 0.086978 | -0.814960 | 0 | 0 | 10 | 10 |
