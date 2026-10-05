# Results: partial release-split reproduction

All tables are derived from structured run artifacts; no model was selected using test results.

## Task and model coverage

| Task | Model | Runs | Parameters | F1 mean ± SD | AUROC mean ± SD |
|---|---|---:|---:|---:|---:|
| binding | gate_source | 3/3 | 81 | 0.7466 ± 0.0143 | 0.7743 ± 0.0201 |
| binding | gate_corrected | 3/3 | 81 | 0.7577 ± 0.0086 | 0.8350 ± 0.0053 |
| binding | merlin_reference | 3/3 | 346 | 0.7344 ± 0.0407 | 0.7829 ± 0.0355 |
| binding | classical_reference | 3/3 | 350 | 0.7400 ± 0.0076 | 0.8047 ± 0.0088 |
| binding | logistic_onehot | 3/3 | 81 | 0.8064 ± 0.0000 | 0.8881 ± 0.0000 |
| binding | rf_onehot | 3/3 | 300 trees | 0.8031 ± 0.0034 | 0.8785 ± 0.0010 |
| binding | merlin | 3/3 | 256 | 0.7271 ± 0.0282 | 0.7868 ± 0.0224 |
| binding | classical | 3/3 | 250 | 0.7365 ± 0.0342 | 0.8024 ± 0.0159 |
| binding | logistic | 3/3 | 21 | 0.8127 ± 0.0000 | 0.8711 ± 0.0000 |
| binding | rf | 3/3 | 300 trees | 0.8085 ± 0.0054 | 0.8830 ± 0.0024 |
| immunogenic | gate_source | 3/3 | 81 | 0.5120 ± 0.0393 | 0.5515 ± 0.0245 |
| immunogenic | gate_corrected | 3/3 | 81 | 0.5098 ± 0.0148 | 0.5607 ± 0.0170 |
| immunogenic | merlin_reference | 3/3 | 346 | 0.5336 ± 0.0208 | 0.5475 ± 0.0205 |
| immunogenic | classical_reference | 3/3 | 350 | 0.5478 ± 0.0172 | 0.5622 ± 0.0109 |
| immunogenic | logistic_onehot | 3/3 | 81 | 0.6252 ± 0.0000 | 0.6541 ± 0.0000 |
| immunogenic | rf_onehot | 3/3 | 300 trees | 0.6054 ± 0.0047 | 0.6600 ± 0.0050 |
| immunogenic | merlin | 3/3 | 256 | 0.5244 ± 0.0284 | 0.5420 ± 0.0144 |
| immunogenic | classical | 3/3 | 250 | 0.5337 ± 0.0635 | 0.5670 ± 0.0260 |
| immunogenic | logistic | 3/3 | 21 | 0.5554 ± 0.0000 | 0.5757 ± 0.0000 |
| immunogenic | rf | 3/3 | 300 trees | 0.5674 ± 0.0070 | 0.5811 ± 0.0022 |

## Comparison with the paper

The paper reports QRAC-10 binding F1 0.73 and selected binder-only immunogenicity F1 0.67 [1]. This protocol uses the larger released test sets and 100 updates, not the exact historical experiment, so the deltas are descriptive only.

| Task | Paper F1 | Local source F1 | Delta | Agreement |
|---|---:|---:|---:|---|
| binding | 0.73 | 0.7466 | +0.0166 | Similar binding range, not a quantitative replication |
| immunogenic | 0.67 | 0.5120 | -0.1580 | Reported immunogenicity performance not matched |

## Label-convention sensitivity (single seed 49)

| Task | Zero/one source F1 | AUROC | Accuracy |
|---|---:|---:|---:|
| binding | 0.7159 | 0.7343 | 0.6727 |
| immunogenic | 0.6135 | 0.4588 | 0.4860 |

With 0/1 labels the source objective gives negative examples zero loss. The higher immunogenicity F1 in this run comes with AUROC below 0.5, so F1 alone is misleading. This does not establish which labels were used in every published experiment.

## MerLin resource assumptions

| Task / model | Space / detector | Photons / modes per cell | Input state | Postselection | Backend | Shots | Seeds | Mean wall seconds |
|---|---|---|---|---|---|---|---|---:|
| binding / merlin_reference | FOCK / ideal PNR | 2 / 7, 3 cell executions | [1, 0, 0, 1, 0, 0, 0] | none | MerLin 0.4.1 CPU analytic simulator | analytic training; analytic and 1000/cell test | 49, 50, 51 | 22.95 |
| binding / merlin | FOCK / ideal PNR | 2 / 6, 3 cell executions | [1, 0, 0, 1, 0, 0] | none | MerLin 0.4.1 CPU analytic simulator | analytic training; analytic and 1000/cell test | 49, 50, 51 | 37.56 |
| immunogenic / merlin_reference | FOCK / ideal PNR | 2 / 7, 3 cell executions | [1, 0, 0, 1, 0, 0, 0] | none | MerLin 0.4.1 CPU analytic simulator | analytic training; analytic and 1000/cell test | 49, 50, 51 | 26.77 |
| immunogenic / merlin | FOCK / ideal PNR | 2 / 6, 3 cell executions | [1, 0, 0, 1, 0, 0] | none | MerLin 0.4.1 CPU analytic simulator | analytic training; analytic and 1000/cell test | 49, 50, 51 | 34.52 |

## Finite-shot F1

| Task / model | Shots per cell | Shots per peptide | F1 mean ± SD |
|---|---:|---:|---:|
| binding / merlin | 0 | 0 | 0.7271 ± 0.0282 |
| binding / merlin | 1000 | 3000 | 0.7271 ± 0.0251 |
| binding / merlin_reference | 0 | 0 | 0.7344 ± 0.0407 |
| binding / merlin_reference | 1000 | 3000 | 0.7246 ± 0.0448 |
| immunogenic / merlin | 0 | 0 | 0.5244 ± 0.0284 |
| immunogenic / merlin | 1000 | 3000 | 0.5254 ± 0.0344 |
| immunogenic / merlin_reference | 0 | 0 | 0.5336 ± 0.0208 |
| immunogenic / merlin_reference | 1000 | 3000 | 0.5344 ± 0.0201 |

## Interpretation and limitations

The seven-mode reference adaptation fixes the global-phase feature alias of the legacy six-mode model, but shows no clear advantage over its 350-parameter matched MLP. Legacy six-mode results are kept as a labelled ablation. The 81-parameter one-hot logistic control is stronger than the 81-parameter source QCNN on both tasks.

This is not a hardware-speed or clinical-utility claim. Three seeds measure optimisation variation on one split, not independent-cohort uncertainty. One binding test row without assay truth is excluded; the released 1000-row test sets differ from the paper's 50/100 descriptions. Experimental assay truth follows the supplement legend [12].

## Supplied affinity-rank diagnostic (no fitting)

| Population | Test count | AUROC (negative BA rank) | F1 (rank < 2) |
|---|---:|---:|---:|
| binding | 999 | 0.9560 | 0.8671 |
| mixed | 1000 | 0.7611 | 0.6842 |
| immunogenic | 1000 | 0.3342 | 0.5059 |

This evaluates the supplied, externally trained affinity predictions; it is not a parameter-matched baseline. The ranking association changes sharply after conditioning on experimental binders.

## Sources

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13398535
[12]: https://pmc.ncbi.nlm.nih.gov/articles/instance/13398535/bin/sciadv.aec3824_sm.pdf
