# S7: decisive comparisons on fresh seeds 5-9 (post-handoff independent check)

Metric: test RMSE (AQI), observed targets; lower is better. Seeds 5-9 were never used before. Decision rule predeclared in LOG.md (Post-handoff check S7).

## Fresh seeds 5-9, 80/20 split

Scope: paper-accurate data and split; paper hp (lr 1e-3) and S3-tuned lr (1e-2) for the LSTMs; 5 fresh seeds.

| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| Paper LSTM(16), lr 1e-3 | paper baseline | local run (s7_fresh_seeds_20261029-161122:B1_lstm_1873) | paper hp | 5 / 5 | 12.572 | 3.051 | paper architecture | 1873 | 0.3 | COMPLETE |
| LSTM(3), lr 1e-2 | iso-parameter, tuned | local run (s7_fresh_seeds_20261029-161122:B2_lstm3_lr0.01) | S3-tuned lr | 5 / 5 | 13.149 | 3.760 | 196/209 params | 196 | 0.3 | COMPLETE |
| LSTM(3), lr 1e-3 | iso-parameter | local run (s7_fresh_seeds_20261029-161122:B2_lstm_matched_196) | paper hp | 5 / 5 | 13.522 | 1.826 | 196/209 params | 196 | 0.3 | COMPLETE |
| Paper LSTM(16), lr 1e-2 | paper baseline, tuned | local run (s7_fresh_seeds_20261029-161122:B1_lstm16_lr0.01) | S3-tuned lr | 5 / 5 | 13.835 | 3.426 | paper architecture | 1873 | 0.3 | COMPLETE |
| Q-B QK-LSTM (209) | reproduced method | local run (s7_fresh_seeds_20261029-161122:QB_quantum_209) | paper hp | 5 / 5 | 15.826 | 3.394 | reference | 209 | 5.2 | COMPLETE |
| QK-LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 9.20 | — | — | 209 | — | reference only |
| LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 15.94 | — | — | 1873 | — | reference only |

Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):

| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |
|---|---|---|---|---:|---|---:|---|
| (a) C1 paper hp | Q-B QK-LSTM (209) | Paper LSTM(16), lr 1e-3 | COMMENSURABLE | +3.254 | [+0.928, +5.581] | 0/5 | **A worse** |
| (b) C2 paper hp | Q-B QK-LSTM (209) | LSTM(3), lr 1e-3 | COMMENSURABLE | +2.304 | [-0.557, +5.165] | 1/5 | **not separated** |
| (c) C1 tuned | Q-B QK-LSTM (209) | Paper LSTM(16), lr 1e-2 | COMMENSURABLE | +1.992 | [-0.814, +4.797] | 2/5 | **not separated** |
| (d) C2 tuned | Q-B QK-LSTM (209) | LSTM(3), lr 1e-2 | COMMENSURABLE | +2.677 | [+1.573, +3.781] | 0/5 | **A worse** |

Code commits across included runs: ['bd0976559f98f9710419b79a9ce1d1f60205f4b5']

