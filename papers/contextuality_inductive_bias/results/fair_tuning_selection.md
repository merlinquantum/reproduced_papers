### Phase 5 tuning: selection by mean final validation NLL (seeds 0-2)

| Model | lr | L2 | Eligible | Valid / expected | Mean validation NLL | Selected |
|---|---:|---:|---|---:|---:|---|
| `{"--cnot-pattern": "paper", "--layers": 1, "--model": "generic_quantum"}` | 0.001 | 0 | yes | 3/3 | 0.687432 |  |
| `{"--cnot-pattern": "paper", "--layers": 1, "--model": "generic_quantum"}` | 0.003 | 0 | yes | 3/3 | 0.686871 | **yes** |
| `{"--cnot-pattern": "paper", "--layers": 1, "--model": "generic_quantum"}` | 0.01 | 0 | yes | 3/3 | 0.686828 |  |
| `{"--cnot-pattern": "paper", "--layers": 2, "--model": "generic_quantum"}` | 0.001 | 0 | yes | 3/3 | 0.686633 | **yes** |
| `{"--cnot-pattern": "paper", "--layers": 2, "--model": "generic_quantum"}` | 0.003 | 0 | yes | 3/3 | 0.688175 |  |
| `{"--cnot-pattern": "paper", "--layers": 2, "--model": "generic_quantum"}` | 0.01 | 0 | yes | 3/3 | 0.688326 |  |
| `{"--computation-space": "fock", "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0.001 | 0 | yes | 3/3 | 0.686528 |  |
| `{"--computation-space": "fock", "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0.003 | 0 | yes | 3/3 | 0.686409 | **yes** |
| `{"--computation-space": "fock", "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0.01 | 0 | yes | 3/3 | 0.686511 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 0 | yes | 3/3 | 0.698352 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 0.0001 | yes | 3/3 | 0.693902 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 0.001 | yes | 3/3 | 0.688710 | **yes** |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 0.01 | yes | 3/3 | 0.693175 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 0.1 | yes | 3/3 | 0.693149 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.001 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 0 | yes | 3/3 | 0.708668 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 0.0001 | yes | 3/3 | 0.695862 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 0.001 | yes | 3/3 | 0.688713 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 0.01 | yes | 3/3 | 0.693175 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 0.1 | yes | 3/3 | 0.693149 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.003 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 0 | yes | 3/3 | 0.724013 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 0.0001 | yes | 3/3 | 0.698056 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 0.001 | yes | 3/3 | 0.688713 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 0.01 | yes | 3/3 | 0.693175 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 0.1 | yes | 3/3 | 0.693149 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 7, "--model": "mlp_biased"}` | 0.01 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 0 | yes | 3/3 | 0.697313 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 0.0001 | yes | 3/3 | 0.692237 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 0.001 | yes | 3/3 | 0.688854 | **yes** |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 0.01 | yes | 3/3 | 0.693224 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 0.1 | yes | 3/3 | 0.693153 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.001 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 0 | yes | 3/3 | 0.705743 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 0.0001 | yes | 3/3 | 0.696561 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 0.001 | yes | 3/3 | 0.688840 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 0.01 | yes | 3/3 | 0.693224 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 0.1 | yes | 3/3 | 0.693152 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.003 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 0 | yes | 3/3 | 0.709434 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 0.0001 | yes | 3/3 | 0.696332 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 0.001 | yes | 3/3 | 0.688844 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 0.01 | yes | 3/3 | 0.693224 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 0.1 | yes | 3/3 | 0.693153 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 1 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 10 | yes | 3/3 | 0.693147 |  |
| `{"--hidden": 9, "--model": "mlp_unbiased"}` | 0.01 | 100 | yes | 3/3 | 0.693147 |  |
| `{"--layers": 1, "--model": "biased_quantum"}` | 0.001 | 0 | yes | 3/3 | 0.684974 | **yes** |
| `{"--layers": 1, "--model": "biased_quantum"}` | 0.003 | 0 | yes | 3/3 | 0.685120 |  |
| `{"--layers": 1, "--model": "biased_quantum"}` | 0.01 | 0 | yes | 3/3 | 0.685074 |  |
| `{"--layers": 2, "--model": "biased_quantum"}` | 0.001 | 0 | yes | 3/3 | 0.686644 | **yes** |
| `{"--layers": 2, "--model": "biased_quantum"}` | 0.003 | 0 | yes | 3/3 | 0.687669 |  |
| `{"--layers": 2, "--model": "biased_quantum"}` | 0.01 | 0 | yes | 3/3 | 0.687576 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 0 | yes | 3/3 | 0.749509 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 0.01 | yes | 3/3 | 0.688673 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 0.1 | yes | 3/3 | 0.684874 | **yes** |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 1 | yes | 3/3 | 0.688131 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 10 | yes | 3/3 | 0.692135 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.001 | 100 | yes | 3/3 | 0.693163 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 0 | yes | 3/3 | 0.726555 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 0.01 | yes | 3/3 | 0.687343 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 0.1 | yes | 3/3 | 0.686617 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 1 | yes | 3/3 | 0.688130 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 10 | yes | 3/3 | 0.692164 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.003 | 100 | yes | 3/3 | 0.693037 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 0 | yes | 3/3 | 0.740361 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 0.01 | yes | 3/3 | 0.709777 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 0.1 | yes | 3/3 | 0.824604 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 1 | yes | 3/3 | 0.688131 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 10 | yes | 3/3 | 0.692159 |  |
| `{"--model": "bias_fourier", "--spectrum": 1}` | 0.01 | 100 | yes | 3/3 | 0.694799 |  |
| `{"--model": "quadratic"}` | 0.001 | 0 | yes | 3/3 | 0.692093 |  |
| `{"--model": "quadratic"}` | 0.001 | 0.0001 | yes | 3/3 | 0.691501 |  |
| `{"--model": "quadratic"}` | 0.001 | 0.001 | yes | 3/3 | 0.689430 |  |
| `{"--model": "quadratic"}` | 0.001 | 0.01 | yes | 3/3 | 0.687901 | **yes** |
| `{"--model": "quadratic"}` | 0.001 | 0.1 | yes | 3/3 | 0.690267 |  |
| `{"--model": "quadratic"}` | 0.001 | 1 | yes | 3/3 | 0.692657 |  |
| `{"--model": "quadratic"}` | 0.001 | 10 | yes | 3/3 | 0.693093 |  |
| `{"--model": "quadratic"}` | 0.001 | 100 | yes | 3/3 | 0.693142 |  |
| `{"--model": "quadratic"}` | 0.003 | 0 | yes | 3/3 | 0.692387 |  |
| `{"--model": "quadratic"}` | 0.003 | 0.0001 | yes | 3/3 | 0.691560 |  |
| `{"--model": "quadratic"}` | 0.003 | 0.001 | yes | 3/3 | 0.689321 |  |
| `{"--model": "quadratic"}` | 0.003 | 0.01 | yes | 3/3 | 0.687900 |  |
| `{"--model": "quadratic"}` | 0.003 | 0.1 | yes | 3/3 | 0.690267 |  |
| `{"--model": "quadratic"}` | 0.003 | 1 | yes | 3/3 | 0.692657 |  |
| `{"--model": "quadratic"}` | 0.003 | 10 | yes | 3/3 | 0.693093 |  |
| `{"--model": "quadratic"}` | 0.003 | 100 | yes | 3/3 | 0.693142 |  |
| `{"--model": "quadratic"}` | 0.01 | 0 | yes | 3/3 | 0.692484 |  |
| `{"--model": "quadratic"}` | 0.01 | 0.0001 | yes | 3/3 | 0.691564 |  |
| `{"--model": "quadratic"}` | 0.01 | 0.001 | yes | 3/3 | 0.689316 |  |
| `{"--model": "quadratic"}` | 0.01 | 0.01 | yes | 3/3 | 0.687900 |  |
| `{"--model": "quadratic"}` | 0.01 | 0.1 | yes | 3/3 | 0.690267 |  |
| `{"--model": "quadratic"}` | 0.01 | 1 | yes | 3/3 | 0.692657 |  |
| `{"--model": "quadratic"}` | 0.01 | 10 | yes | 3/3 | 0.693094 |  |
| `{"--model": "quadratic"}` | 0.01 | 100 | yes | 3/3 | 0.693142 |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 0 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 0.01 | yes | 3/3 | 0.691898 |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 0.1 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 1 | yes | 3/3 | 0.685958 | **yes** |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 10 | yes | 3/3 | 0.689399 |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.001 | 100 | yes | 3/3 | 0.692594 |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 0 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 0.01 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 0.1 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 1 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 10 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.003 | 100 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 0 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 0.01 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 0.1 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 1 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 10 | no | 0/3 | n/a |  |
| `{"--model": "surrogate", "--spectrum": 1}` | 0.01 | 100 | no | 0/3 | n/a |  |
