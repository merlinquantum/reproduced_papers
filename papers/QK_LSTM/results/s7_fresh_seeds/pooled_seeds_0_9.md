# Pooled seeds 0-9 (S1/S4 seeds 0-4 + S7 seeds 5-9) — supplementary, NOT pre-declared

Q-B QK-LSTM minus baseline, test RMSE (AQI), paired by seed; two-sided 95% t-interval, n = 10. Exploratory: the predeclared S7 decision uses seeds 5-9 only.

| Comparison | n | Mean Δ | 95% CI | QK better on | Decision |
|---|---:|---:|---|---:|---|
| (a) vs paper LSTM, paper hp | 10 | +3.16 | [+1.90, +4.42] | 0/10 | worse |
| (b) vs LSTM(3), paper hp | 10 | +2.32 | [+1.18, +3.45] | 1/10 | worse |
| (c) vs paper LSTM, tuned lr 1e-2 | 10 | +2.39 | [+0.97, +3.82] | 2/10 | worse |
| (d) vs LSTM(3), tuned lr 1e-2 | 10 | +3.32 | [+2.13, +4.51] | 0/10 | worse |
