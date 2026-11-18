### Fig. 8: reproduced vs paper (paper values read off the figure, about +-5% on KL)

| Protocol | Reading | Paper curve | Metric | Paper | Ours (mean +- std, n) | Rel. diff |
|---|---|---|---|---:|---|---:|
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.004606 +- 0.0003 (n=5) | +4.7% |
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.004508 +- 0.0003 (n=5) | +4.8% |
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.6815 +- 0.0004 (n=5) | +0.1% |
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.007645 +- 0.0008 (n=5) | -19.5% |
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.007181 +- 0.0009 (n=5) | -22.8% |
| code | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6799 +- 0.0006 (n=5) | -0.7% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.007481 +- 2e-05 (n=5) | -1.6% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.006231 +- 2e-05 (n=5) | -1.1% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6765 +- 7e-06 (n=5) | +0.1% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.01271 +- 3e-06 (n=5) | +1.7% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.006945 +- 1e-05 (n=5) | -3.5% |
| code | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6588 +- 3e-06 (n=5) | -0.0% |
| code | author code (L=1, code CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.004675 +- 0.0001 (n=5) | +6.2% |
| code | author code (L=1, code CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.00466 +- 0.0001 (n=5) | +8.4% |
| code | author code (L=1, code CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.6839 +- 0.0006 (n=5) | +0.4% |
| code | author code (L=1, code CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.009271 +- 0.0004 (n=5) | -2.4% |
| code | author code (L=1, code CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.008796 +- 0.0007 (n=5) | -5.4% |
| code | author code (L=1, code CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6867 +- 0.001 (n=5) | +0.2% |
| code | author code (L=1, code CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.007481 +- 2e-05 (n=5) | -1.6% |
| code | author code (L=1, code CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.006231 +- 2e-05 (n=5) | -1.1% |
| code | author code (L=1, code CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6765 +- 7e-06 (n=5) | +0.1% |
| code | author code (L=1, code CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.01271 +- 3e-06 (n=5) | +1.7% |
| code | author code (L=1, code CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.006945 +- 1e-05 (n=5) | -3.5% |
| code | author code (L=1, code CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6588 +- 3e-06 (n=5) | -0.0% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.005144 +- 0.0006 (n=20) | +16.9% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.004675 +- 0.0005 (n=20) | +8.7% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.6811 +- 0.0006 (n=20) | +0.0% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.007023 +- 0.0007 (n=20) | -26.1% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.006784 +- 0.0006 (n=20) | -27.1% |
| paper | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6794 +- 0.001 (n=20) | -0.8% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.005448 +- 6e-06 (n=5) | -28.3% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.004473 +- 1e-05 (n=5) | -29.0% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6765 +- 7e-06 (n=5) | +0.1% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.01865 +- 4e-06 (n=5) | +49.2% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.004991 +- 4e-06 (n=5) | -30.7% |
| paper | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6588 +- 3e-06 (n=5) | -0.0% |
| paper | author code (L=1, code CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.00408 +- 0.0003 (n=20) | -7.3% |
| paper | author code (L=1, code CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.004026 +- 0.0003 (n=20) | -6.4% |
| paper | author code (L=1, code CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.6839 +- 0.0006 (n=20) | +0.4% |
| paper | author code (L=1, code CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.007231 +- 0.0008 (n=20) | -23.9% |
| paper | author code (L=1, code CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.007195 +- 0.0008 (n=20) | -22.6% |
| paper | author code (L=1, code CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6865 +- 0.001 (n=20) | +0.2% |
| paper | author code (L=1, code CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.005448 +- 6e-06 (n=5) | -28.3% |
| paper | author code (L=1, code CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.004473 +- 1e-05 (n=5) | -29.0% |
| paper | author code (L=1, code CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6765 +- 7e-06 (n=5) | +0.1% |
| paper | author code (L=1, code CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.01865 +- 4e-06 (n=5) | +49.2% |
| paper | author code (L=1, code CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.004991 +- 4e-06 (n=5) | -30.7% |
| paper | author code (L=1, code CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6588 +- 3e-06 (n=5) | -0.0% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.003604 +- 0.0002 (n=5) | -18.1% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.003511 +- 0.0001 (n=5) | -18.3% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.6833 +- 0.0007 (n=5) | +0.3% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.00483 +- 0.0002 (n=5) | -49.2% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.00483 +- 0.0002 (n=5) | -48.1% |
| fork | paper text (L=2, Fig. 10 CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6825 +- 0.0008 (n=5) | -0.4% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.004276 +- 6e-06 (n=5) | -43.7% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.004276 +- 6e-06 (n=5) | -32.1% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6824 +- 9e-06 (n=5) | +0.9% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.006965 +- 2e-06 (n=5) | -44.3% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.004174 +- 1e-06 (n=5) | -42.0% |
| fork | paper text (L=2, Fig. 10 CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6752 +- 6e-07 (n=5) | +2.5% |
| fork | author code (L=1, code CNOTs) | quantum (biased) | final test KL | 0.0044 | 0.003797 +- 0.0002 (n=5) | -13.7% |
| fork | author code (L=1, code CNOTs) | quantum (biased) | min test KL | 0.0043 | 0.003797 +- 0.0002 (n=5) | -11.7% |
| fork | author code (L=1, code CNOTs) | quantum (biased) | final train NLL | 0.681 | 0.685 +- 0.0004 (n=5) | +0.6% |
| fork | author code (L=1, code CNOTs) | quantum (no bias) | final test KL | 0.0095 | 0.006628 +- 0.001 (n=5) | -30.2% |
| fork | author code (L=1, code CNOTs) | quantum (no bias) | min test KL | 0.0093 | 0.006628 +- 0.001 (n=5) | -28.7% |
| fork | author code (L=1, code CNOTs) | quantum (no bias) | final train NLL | 0.685 | 0.6871 +- 0.0009 (n=5) | +0.3% |
| fork | author code (L=1, code CNOTs) | surrogate L=1 | final test KL | 0.0076 | 0.004276 +- 6e-06 (n=5) | -43.7% |
| fork | author code (L=1, code CNOTs) | surrogate L=1 | min test KL | 0.0063 | 0.004276 +- 6e-06 (n=5) | -32.1% |
| fork | author code (L=1, code CNOTs) | surrogate L=1 | final train NLL | 0.676 | 0.6824 +- 9e-06 (n=5) | +0.9% |
| fork | author code (L=1, code CNOTs) | surrogate L=2 | final test KL | 0.0125 | 0.006965 +- 2e-06 (n=5) | -44.3% |
| fork | author code (L=1, code CNOTs) | surrogate L=2 | min test KL | 0.0072 | 0.004174 +- 1e-06 (n=5) | -42.0% |
| fork | author code (L=1, code CNOTs) | surrogate L=2 | final train NLL | 0.659 | 0.6752 +- 6e-07 (n=5) | +2.5% |
