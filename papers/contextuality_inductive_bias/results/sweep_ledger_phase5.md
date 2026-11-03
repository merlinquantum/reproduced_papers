#### `fair_tuning`

- Status FAILED; 2026-10-16T20:16:35Z to 2026-10-16T22:34:33Z; commit e7beb6a4f0 (dirty=True); paper_code_sha256 806768297e1f (unchanged)
- Coverage: PARTIAL SWEEP. Planned 180, expected 180, completed 141, duplicates 0, failed 39, runs with a retry 39
- Summary: `None`. Command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/simon/Documents/Reproduced papers/reproduced_papers/implementation.py --paper contextuality_inductive_bias --config configs/fair_quantum_tuning.json --outdir /Users/simon/Documents/Reproduced papers/reproduced_papers/papers/contextuality_inductive_bias/outdir/sweeps/fair_tuning/runs/fair_tuning__c000_r00/attempt1 --sweep-id fair_tuning ...`

| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |
|---|---|---:|---:|---|---:|---|
| fair_tuning__c000_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 162.0 | `runs/fair_tuning__c000_r00/attempt1/run_20261016-221638` |
| fair_tuning__c000_r01 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 162.6 | `runs/fair_tuning__c000_r01/attempt1/run_20261016-221638` |
| fair_tuning__c000_r02 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 162.0 | `runs/fair_tuning__c000_r02/attempt1/run_20261016-221638` |
| fair_tuning__c001_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 251.3 | `runs/fair_tuning__c001_r00/attempt1/run_20261016-221638` |
| fair_tuning__c001_r01 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 253.6 | `runs/fair_tuning__c001_r01/attempt1/run_20261016-221638` |
| fair_tuning__c001_r02 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 253.5 | `runs/fair_tuning__c001_r02/attempt1/run_20261016-221638` |
| fair_tuning__c002_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 146.5 | `runs/fair_tuning__c002_r00/attempt1/run_20261016-221921` |
| fair_tuning__c002_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 146.5 | `runs/fair_tuning__c002_r01/attempt1/run_20261016-221921` |
| fair_tuning__c002_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 146.9 | `runs/fair_tuning__c002_r02/attempt1/run_20261016-221922` |
| fair_tuning__c003_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 249.5 | `runs/fair_tuning__c003_r00/attempt1/run_20261016-222051` |
| fair_tuning__c003_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 244.6 | `runs/fair_tuning__c003_r01/attempt1/run_20261016-222053` |
| fair_tuning__c003_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 248.0 | `runs/fair_tuning__c003_r02/attempt1/run_20261016-222053` |
| fair_tuning__c004_r00 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 14.0 | `runs/fair_tuning__c004_r00/attempt1/run_20261016-222148` |
| fair_tuning__c004_r01 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 14.0 | `runs/fair_tuning__c004_r01/attempt1/run_20261016-222148` |
| fair_tuning__c004_r02 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 14.4 | `runs/fair_tuning__c004_r02/attempt1/run_20261016-222149` |
| fair_tuning__c005_r00 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 13.6 | `runs/fair_tuning__c005_r00/attempt1/run_20261016-222202` |
| fair_tuning__c005_r01 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 13.4 | `runs/fair_tuning__c005_r01/attempt1/run_20261016-222202` |
| fair_tuning__c005_r02 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 14.0 | `runs/fair_tuning__c005_r02/attempt1/run_20261016-222204` |
| fair_tuning__c006_r00 | `{"--l2": 0.0, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 13.6 | `runs/fair_tuning__c006_r00/attempt1/run_20261016-222215` |
| fair_tuning__c006_r01 | `{"--l2": 0.0, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 13.4 | `runs/fair_tuning__c006_r01/attempt1/run_20261016-222215` |
| fair_tuning__c006_r02 | `{"--l2": 0.0, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 14.3 | `runs/fair_tuning__c006_r02/attempt1/run_20261016-222218` |
| fair_tuning__c007_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.001, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0 | 1 | COMPLETED | 543.9 | `runs/fair_tuning__c007_r00/attempt1/run_20261016-222229` |
| fair_tuning__c007_r01 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.001, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 1 | 1 | COMPLETED | 542.1 | `runs/fair_tuning__c007_r01/attempt1/run_20261016-222229` |
| fair_tuning__c007_r02 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.001, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 2 | 1 | COMPLETED | 542.8 | `runs/fair_tuning__c007_r02/attempt1/run_20261016-222233` |
| fair_tuning__c008_r00 | `{"--l2": 0.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 1 | COMPLETED | 260.0 | `runs/fair_tuning__c008_r00/attempt1/run_20261016-222459` |
| fair_tuning__c008_r01 | `{"--l2": 0.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 1 | COMPLETED | 264.3 | `runs/fair_tuning__c008_r01/attempt1/run_20261016-222501` |
| fair_tuning__c008_r02 | `{"--l2": 0.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 1 | COMPLETED | 270.6 | `runs/fair_tuning__c008_r02/attempt1/run_20261016-222502` |
| fair_tuning__c009_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 1 | COMPLETED | 254.4 | `runs/fair_tuning__c009_r00/attempt1/run_20261016-222920` |
| fair_tuning__c009_r01 | `{"--l2": 0.01, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 1 | COMPLETED | 253.9 | `runs/fair_tuning__c009_r01/attempt1/run_20261016-222927` |
| fair_tuning__c009_r02 | `{"--l2": 0.01, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 1 | COMPLETED | 264.7 | `runs/fair_tuning__c009_r02/attempt1/run_20261016-222934` |
| fair_tuning__c010_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 248.9 | `None` |
| fair_tuning__c010_r01 | `{"--l2": 0.1, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 242.9 | `None` |
| fair_tuning__c010_r02 | `{"--l2": 0.1, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 203.2 | `None` |
| fair_tuning__c011_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 1 | COMPLETED | 258.4 | `runs/fair_tuning__c011_r00/attempt1/run_20261016-223333` |
| fair_tuning__c011_r01 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 1 | COMPLETED | 257.9 | `runs/fair_tuning__c011_r01/attempt1/run_20261016-223341` |
| fair_tuning__c011_r02 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 1 | COMPLETED | 256.8 | `runs/fair_tuning__c011_r02/attempt1/run_20261016-223359` |
| fair_tuning__c012_r00 | `{"--l2": 10.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 1 | COMPLETED | 258.7 | `runs/fair_tuning__c012_r00/attempt1/run_20261016-223753` |
| fair_tuning__c012_r01 | `{"--l2": 10.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 1 | COMPLETED | 260.8 | `runs/fair_tuning__c012_r01/attempt1/run_20261016-223758` |
| fair_tuning__c012_r02 | `{"--l2": 10.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 1 | COMPLETED | 258.3 | `runs/fair_tuning__c012_r02/attempt1/run_20261016-223815` |
| fair_tuning__c013_r00 | `{"--l2": 100.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 0 | 1 | COMPLETED | 258.9 | `runs/fair_tuning__c013_r00/attempt1/run_20261016-223831` |
| fair_tuning__c013_r01 | `{"--l2": 100.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 1 | 1 | COMPLETED | 260.7 | `runs/fair_tuning__c013_r01/attempt1/run_20261016-223958` |
| fair_tuning__c013_r02 | `{"--l2": 100.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 2 | 1 | COMPLETED | 261.1 | `runs/fair_tuning__c013_r02/attempt1/run_20261016-224004` |
| fair_tuning__c014_r00 | `{"--l2": 0.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 564.4 | `runs/fair_tuning__c014_r00/attempt1/run_20261016-224212` |
| fair_tuning__c014_r01 | `{"--l2": 0.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 543.7 | `runs/fair_tuning__c014_r01/attempt1/run_20261016-224220` |
| fair_tuning__c014_r02 | `{"--l2": 0.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 564.5 | `runs/fair_tuning__c014_r02/attempt1/run_20261016-224234` |
| fair_tuning__c015_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 545.3 | `runs/fair_tuning__c015_r00/attempt1/run_20261016-224250` |
| fair_tuning__c015_r01 | `{"--l2": 0.01, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 564.2 | `runs/fair_tuning__c015_r01/attempt1/run_20261016-224419` |
| fair_tuning__c015_r02 | `{"--l2": 0.01, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 545.9 | `runs/fair_tuning__c015_r02/attempt1/run_20261016-224425` |
| fair_tuning__c016_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 552.2 | `runs/fair_tuning__c016_r00/attempt1/run_20261016-225122` |
| fair_tuning__c016_r01 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 555.3 | `runs/fair_tuning__c016_r01/attempt1/run_20261016-225136` |
| fair_tuning__c016_r02 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 569.7 | `runs/fair_tuning__c016_r02/attempt1/run_20261016-225155` |
| fair_tuning__c017_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 574.6 | `runs/fair_tuning__c017_r00/attempt1/run_20261016-225158` |
| fair_tuning__c017_r01 | `{"--l2": 1.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 559.6 | `runs/fair_tuning__c017_r01/attempt1/run_20261016-225331` |
| fair_tuning__c017_r02 | `{"--l2": 1.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 559.7 | `runs/fair_tuning__c017_r02/attempt1/run_20261016-225344` |
| fair_tuning__c018_r00 | `{"--l2": 10.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 566.4 | `runs/fair_tuning__c018_r00/attempt1/run_20261016-230035` |
| fair_tuning__c018_r01 | `{"--l2": 10.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 561.0 | `runs/fair_tuning__c018_r01/attempt1/run_20261016-230051` |
| fair_tuning__c018_r02 | `{"--l2": 10.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 576.1 | `runs/fair_tuning__c018_r02/attempt1/run_20261016-230125` |
| fair_tuning__c019_r00 | `{"--l2": 100.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 577.3 | `runs/fair_tuning__c019_r00/attempt1/run_20261016-230132` |
| fair_tuning__c019_r01 | `{"--l2": 100.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 547.2 | `runs/fair_tuning__c019_r01/attempt1/run_20261016-230250` |
| fair_tuning__c019_r02 | `{"--l2": 100.0, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 545.4 | `runs/fair_tuning__c019_r02/attempt1/run_20261016-230303` |
| fair_tuning__c020_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 176.6 | `runs/fair_tuning__c020_r00/attempt1/run_20261016-231001` |
| fair_tuning__c020_r01 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 174.7 | `runs/fair_tuning__c020_r01/attempt1/run_20261016-231012` |
| fair_tuning__c020_r02 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 165.9 | `runs/fair_tuning__c020_r02/attempt1/run_20261016-231101` |
| fair_tuning__c021_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 255.0 | `runs/fair_tuning__c021_r00/attempt1/run_20261016-231110` |
| fair_tuning__c021_r01 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 246.9 | `runs/fair_tuning__c021_r01/attempt1/run_20261016-231157` |
| fair_tuning__c021_r02 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 248.8 | `runs/fair_tuning__c021_r02/attempt1/run_20261016-231208` |
| fair_tuning__c022_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 147.7 | `runs/fair_tuning__c022_r00/attempt1/run_20261016-231258` |
| fair_tuning__c022_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 147.5 | `runs/fair_tuning__c022_r01/attempt1/run_20261016-231307` |
| fair_tuning__c022_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 143.8 | `runs/fair_tuning__c022_r02/attempt1/run_20261016-231347` |
| fair_tuning__c023_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 274.5 | `runs/fair_tuning__c023_r00/attempt1/run_20261016-231540` |
| fair_tuning__c023_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 270.9 | `runs/fair_tuning__c023_r01/attempt1/run_20261016-231540` |
| fair_tuning__c023_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.003, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 266.1 | `runs/fair_tuning__c023_r02/attempt1/run_20261016-231540` |
| fair_tuning__c024_r00 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 15.7 | `runs/fair_tuning__c024_r00/attempt1/run_20261016-231604` |
| fair_tuning__c024_r01 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 15.2 | `runs/fair_tuning__c024_r01/attempt1/run_20261016-231611` |
| fair_tuning__c024_r02 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 14.7 | `runs/fair_tuning__c024_r02/attempt1/run_20261016-231617` |
| fair_tuning__c025_r00 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 14.2 | `runs/fair_tuning__c025_r00/attempt1/run_20261016-231619` |
| fair_tuning__c025_r01 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 14.2 | `runs/fair_tuning__c025_r01/attempt1/run_20261016-231626` |
| fair_tuning__c025_r02 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 14.3 | `runs/fair_tuning__c025_r02/attempt1/run_20261016-231631` |
| fair_tuning__c026_r00 | `{"--l2": 0.0, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 13.8 | `runs/fair_tuning__c026_r00/attempt1/run_20261016-231633` |
| fair_tuning__c026_r01 | `{"--l2": 0.0, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 14.2 | `runs/fair_tuning__c026_r01/attempt1/run_20261016-231640` |
| fair_tuning__c026_r02 | `{"--l2": 0.0, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 14.5 | `runs/fair_tuning__c026_r02/attempt1/run_20261016-231646` |
| fair_tuning__c027_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0 | 1 | COMPLETED | 504.9 | `runs/fair_tuning__c027_r00/attempt1/run_20261016-231647` |
| fair_tuning__c027_r01 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 1 | 1 | COMPLETED | 505.3 | `runs/fair_tuning__c027_r01/attempt1/run_20261016-231655` |
| fair_tuning__c027_r02 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 2 | 1 | COMPLETED | 503.0 | `runs/fair_tuning__c027_r02/attempt1/run_20261016-231701` |
| fair_tuning__c028_r00 | `{"--l2": 0.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 7.5 | `None` |
| fair_tuning__c028_r01 | `{"--l2": 0.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 7.6 | `None` |
| fair_tuning__c028_r02 | `{"--l2": 0.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 7.7 | `None` |
| fair_tuning__c029_r00 | `{"--l2": 0.01, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 7.2 | `None` |
| fair_tuning__c029_r01 | `{"--l2": 0.01, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 7.1 | `None` |
| fair_tuning__c029_r02 | `{"--l2": 0.01, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c030_r00 | `{"--l2": 0.1, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 7.2 | `None` |
| fair_tuning__c030_r01 | `{"--l2": 0.1, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 7.2 | `None` |
| fair_tuning__c030_r02 | `{"--l2": 0.1, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 7.2 | `None` |
| fair_tuning__c031_r00 | `{"--l2": 1.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 7.2 | `None` |
| fair_tuning__c031_r01 | `{"--l2": 1.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c031_r02 | `{"--l2": 1.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c032_r00 | `{"--l2": 10.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c032_r01 | `{"--l2": 10.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c032_r02 | `{"--l2": 10.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 7.3 | `None` |
| fair_tuning__c033_r00 | `{"--l2": 100.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 121.8 | `None` |
| fair_tuning__c033_r01 | `{"--l2": 100.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 118.6 | `None` |
| fair_tuning__c033_r02 | `{"--l2": 100.0, "--lr": 0.003, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 121.2 | `None` |
| fair_tuning__c034_r00 | `{"--l2": 0.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 557.4 | `runs/fair_tuning__c034_r00/attempt1/run_20261016-232510` |
| fair_tuning__c034_r01 | `{"--l2": 0.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 541.0 | `runs/fair_tuning__c034_r01/attempt1/run_20261016-232510` |
| fair_tuning__c034_r02 | `{"--l2": 0.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 557.0 | `runs/fair_tuning__c034_r02/attempt1/run_20261016-232512` |
| fair_tuning__c035_r00 | `{"--l2": 0.01, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 543.3 | `runs/fair_tuning__c035_r00/attempt1/run_20261016-232513` |
| fair_tuning__c035_r01 | `{"--l2": 0.01, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 560.1 | `runs/fair_tuning__c035_r01/attempt1/run_20261016-232520` |
| fair_tuning__c035_r02 | `{"--l2": 0.01, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 560.7 | `runs/fair_tuning__c035_r02/attempt1/run_20261016-232524` |
| fair_tuning__c036_r00 | `{"--l2": 0.1, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 559.7 | `runs/fair_tuning__c036_r00/attempt1/run_20261016-233413` |
| fair_tuning__c036_r01 | `{"--l2": 0.1, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 545.2 | `runs/fair_tuning__c036_r01/attempt1/run_20261016-233418` |
| fair_tuning__c036_r02 | `{"--l2": 0.1, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 561.0 | `runs/fair_tuning__c036_r02/attempt1/run_20261016-233429` |
| fair_tuning__c037_r00 | `{"--l2": 1.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 550.4 | `runs/fair_tuning__c037_r00/attempt1/run_20261016-233430` |
| fair_tuning__c037_r01 | `{"--l2": 1.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 559.4 | `runs/fair_tuning__c037_r01/attempt1/run_20261016-233440` |
| fair_tuning__c037_r02 | `{"--l2": 1.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 545.8 | `runs/fair_tuning__c037_r02/attempt1/run_20261016-233445` |
| fair_tuning__c038_r00 | `{"--l2": 10.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 547.8 | `runs/fair_tuning__c038_r00/attempt1/run_20261016-234323` |
| fair_tuning__c038_r01 | `{"--l2": 10.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 548.3 | `runs/fair_tuning__c038_r01/attempt1/run_20261016-234333` |
| fair_tuning__c038_r02 | `{"--l2": 10.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 555.1 | `runs/fair_tuning__c038_r02/attempt1/run_20261016-234341` |
| fair_tuning__c039_r00 | `{"--l2": 100.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 545.7 | `runs/fair_tuning__c039_r00/attempt1/run_20261016-234349` |
| fair_tuning__c039_r01 | `{"--l2": 100.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 551.7 | `runs/fair_tuning__c039_r01/attempt1/run_20261016-234350` |
| fair_tuning__c039_r02 | `{"--l2": 100.0, "--lr": 0.003, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 552.7 | `runs/fair_tuning__c039_r02/attempt1/run_20261016-234400` |
| fair_tuning__c040_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 183.2 | `runs/fair_tuning__c040_r00/attempt1/run_20261016-235254` |
| fair_tuning__c040_r01 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 172.6 | `runs/fair_tuning__c040_r01/attempt1/run_20261016-235254` |
| fair_tuning__c040_r02 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 158.1 | `runs/fair_tuning__c040_r02/attempt1/run_20261016-235254` |
| fair_tuning__c041_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "biased_quantum"}` | 0 | 1 | COMPLETED | 244.9 | `runs/fair_tuning__c041_r00/attempt1/run_20261016-235254` |
| fair_tuning__c041_r01 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "biased_quantum"}` | 1 | 1 | COMPLETED | 249.3 | `runs/fair_tuning__c041_r01/attempt1/run_20261016-235302` |
| fair_tuning__c041_r02 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "biased_quantum"}` | 2 | 1 | COMPLETED | 245.8 | `runs/fair_tuning__c041_r02/attempt1/run_20261016-235312` |
| fair_tuning__c042_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 143.8 | `runs/fair_tuning__c042_r00/attempt1/run_20261016-235531` |
| fair_tuning__c042_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 142.1 | `runs/fair_tuning__c042_r01/attempt1/run_20261016-235531` |
| fair_tuning__c042_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.01, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 142.4 | `runs/fair_tuning__c042_r02/attempt1/run_20261016-235532` |
| fair_tuning__c043_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "generic_quantum"}` | 0 | 1 | COMPLETED | 242.3 | `runs/fair_tuning__c043_r00/attempt1/run_20261016-235701` |
| fair_tuning__c043_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "generic_quantum"}` | 1 | 1 | COMPLETED | 241.1 | `runs/fair_tuning__c043_r01/attempt1/run_20261016-235712` |
| fair_tuning__c043_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.01, "--model": "generic_quantum"}` | 2 | 1 | COMPLETED | 239.7 | `runs/fair_tuning__c043_r02/attempt1/run_20261016-235718` |
| fair_tuning__c044_r00 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 14.2 | `runs/fair_tuning__c044_r00/attempt1/run_20261016-235754` |
| fair_tuning__c044_r01 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 14.1 | `runs/fair_tuning__c044_r01/attempt1/run_20261016-235755` |
| fair_tuning__c044_r02 | `{"--hidden": 7, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 14.1 | `runs/fair_tuning__c044_r02/attempt1/run_20261016-235756` |
| fair_tuning__c045_r00 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 13.6 | `runs/fair_tuning__c045_r00/attempt1/run_20261016-235808` |
| fair_tuning__c045_r01 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 13.6 | `runs/fair_tuning__c045_r01/attempt1/run_20261016-235809` |
| fair_tuning__c045_r02 | `{"--hidden": 9, "--l2": 0.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 13.7 | `runs/fair_tuning__c045_r02/attempt1/run_20261016-235810` |
| fair_tuning__c046_r00 | `{"--l2": 0.0, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 13.7 | `runs/fair_tuning__c046_r00/attempt1/run_20261016-235822` |
| fair_tuning__c046_r01 | `{"--l2": 0.0, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 13.8 | `runs/fair_tuning__c046_r01/attempt1/run_20261016-235823` |
| fair_tuning__c046_r02 | `{"--l2": 0.0, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 13.8 | `runs/fair_tuning__c046_r02/attempt1/run_20261016-235824` |
| fair_tuning__c047_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.01, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 0 | 1 | COMPLETED | 478.7 | `runs/fair_tuning__c047_r00/attempt1/run_20261016-235836` |
| fair_tuning__c047_r01 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.01, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 1 | 1 | COMPLETED | 481.2 | `runs/fair_tuning__c047_r01/attempt1/run_20261016-235836` |
| fair_tuning__c047_r02 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.01, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 2 | 1 | COMPLETED | 479.7 | `runs/fair_tuning__c047_r02/attempt1/run_20261016-235837` |
| fair_tuning__c048_r00 | `{"--l2": 0.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 5.8 | `None` |
| fair_tuning__c048_r01 | `{"--l2": 0.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 5.8 | `None` |
| fair_tuning__c048_r02 | `{"--l2": 0.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c049_r00 | `{"--l2": 0.01, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 5.8 | `None` |
| fair_tuning__c049_r01 | `{"--l2": 0.01, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c049_r02 | `{"--l2": 0.01, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c050_r00 | `{"--l2": 0.1, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c050_r01 | `{"--l2": 0.1, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c050_r02 | `{"--l2": 0.1, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 5.8 | `None` |
| fair_tuning__c051_r00 | `{"--l2": 1.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c051_r01 | `{"--l2": 1.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 5.8 | `None` |
| fair_tuning__c051_r02 | `{"--l2": 1.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 5.9 | `None` |
| fair_tuning__c052_r00 | `{"--l2": 10.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 6.0 | `None` |
| fair_tuning__c052_r01 | `{"--l2": 10.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 6.0 | `None` |
| fair_tuning__c052_r02 | `{"--l2": 10.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 6.0 | `None` |
| fair_tuning__c053_r00 | `{"--l2": 100.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 0 | 2 | FAILED | 133.2 | `None` |
| fair_tuning__c053_r01 | `{"--l2": 100.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 1 | 2 | FAILED | 131.9 | `None` |
| fair_tuning__c053_r02 | `{"--l2": 100.0, "--lr": 0.01, "--model": "surrogate", "--spectrum": 1}` | 2 | 2 | FAILED | 9.0 | `None` |
| fair_tuning__c054_r00 | `{"--l2": 0.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 548.5 | `runs/fair_tuning__c054_r00/attempt1/run_20261017-000240` |
| fair_tuning__c054_r01 | `{"--l2": 0.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 554.2 | `runs/fair_tuning__c054_r01/attempt1/run_20261017-000635` |
| fair_tuning__c054_r02 | `{"--l2": 0.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 553.8 | `runs/fair_tuning__c054_r02/attempt1/run_20261017-000637` |
| fair_tuning__c055_r00 | `{"--l2": 0.01, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 559.5 | `runs/fair_tuning__c055_r00/attempt1/run_20261017-000638` |
| fair_tuning__c055_r01 | `{"--l2": 0.01, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 571.7 | `runs/fair_tuning__c055_r01/attempt1/run_20261017-000643` |
| fair_tuning__c055_r02 | `{"--l2": 0.01, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 556.4 | `runs/fair_tuning__c055_r02/attempt1/run_20261017-000644` |
| fair_tuning__c056_r00 | `{"--l2": 0.1, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 583.8 | `runs/fair_tuning__c056_r00/attempt1/run_20261017-001147` |
| fair_tuning__c056_r01 | `{"--l2": 0.1, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 562.1 | `runs/fair_tuning__c056_r01/attempt1/run_20261017-001549` |
| fair_tuning__c056_r02 | `{"--l2": 0.1, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 579.6 | `runs/fair_tuning__c056_r02/attempt1/run_20261017-001552` |
| fair_tuning__c057_r00 | `{"--l2": 1.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 563.4 | `runs/fair_tuning__c057_r00/attempt1/run_20261017-001558` |
| fair_tuning__c057_r01 | `{"--l2": 1.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 579.4 | `runs/fair_tuning__c057_r01/attempt1/run_20261017-001601` |
| fair_tuning__c057_r02 | `{"--l2": 1.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 562.1 | `runs/fair_tuning__c057_r02/attempt1/run_20261017-001615` |
| fair_tuning__c058_r00 | `{"--l2": 10.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 563.3 | `runs/fair_tuning__c058_r00/attempt1/run_20261017-002132` |
| fair_tuning__c058_r01 | `{"--l2": 10.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 542.3 | `runs/fair_tuning__c058_r01/attempt1/run_20261017-002513` |
| fair_tuning__c058_r02 | `{"--l2": 10.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 534.0 | `runs/fair_tuning__c058_r02/attempt1/run_20261017-002522` |
| fair_tuning__c059_r00 | `{"--l2": 100.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 0 | 1 | COMPLETED | 543.2 | `runs/fair_tuning__c059_r00/attempt1/run_20261017-002532` |
| fair_tuning__c059_r01 | `{"--l2": 100.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 1 | 1 | COMPLETED | 534.1 | `runs/fair_tuning__c059_r01/attempt1/run_20261017-002537` |
| fair_tuning__c059_r02 | `{"--l2": 100.0, "--lr": 0.01, "--model": "bias_fourier", "--spectrum": 1}` | 2 | 1 | COMPLETED | 538.1 | `runs/fair_tuning__c059_r02/attempt1/run_20261017-002540` |

#### `fair_tuning_ext`

- Status COMPLETED; 2026-10-16T22:40:33Z to 2026-10-16T22:45:15Z; commit e7beb6a4f0 (dirty=True); paper_code_sha256 b1df8c322a8f (unchanged)
- Coverage: COMPLETE SWEEP. Planned 135, expected 135, completed 135, duplicates 0, failed 0, runs with a retry 0
- Summary: `summary.json`. Command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/simon/Documents/Reproduced papers/reproduced_papers/implementation.py --paper contextuality_inductive_bias --config configs/fair_quantum_tuning.json --outdir /Users/simon/Documents/Reproduced papers/reproduced_papers/papers/contextuality_inductive_bias/outdir/sweeps/fair_tuning_ext/runs/fair_tuning_ext__c000_r00/attempt1 --sweep-id fair_tuning_ext ...`

| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |
|---|---|---:|---:|---|---:|---|
| fair_tuning_ext__c000_r00 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c000_r00/attempt1/run_20261017-004036` |
| fair_tuning_ext__c000_r01 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c000_r01/attempt1/run_20261017-004036` |
| fair_tuning_ext__c000_r02 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c000_r02/attempt1/run_20261017-004036` |
| fair_tuning_ext__c001_r00 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c001_r00/attempt1/run_20261017-004036` |
| fair_tuning_ext__c001_r01 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c001_r01/attempt1/run_20261017-004036` |
| fair_tuning_ext__c001_r02 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c001_r02/attempt1/run_20261017-004036` |
| fair_tuning_ext__c002_r00 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext__c002_r00/attempt1/run_20261017-004049` |
| fair_tuning_ext__c002_r01 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c002_r01/attempt1/run_20261017-004049` |
| fair_tuning_ext__c002_r02 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c002_r02/attempt1/run_20261017-004049` |
| fair_tuning_ext__c003_r00 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c003_r00/attempt1/run_20261017-004049` |
| fair_tuning_ext__c003_r01 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c003_r01/attempt1/run_20261017-004049` |
| fair_tuning_ext__c003_r02 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext__c003_r02/attempt1/run_20261017-004049` |
| fair_tuning_ext__c004_r00 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext__c004_r00/attempt1/run_20261017-004102` |
| fair_tuning_ext__c004_r01 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c004_r01/attempt1/run_20261017-004102` |
| fair_tuning_ext__c004_r02 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c004_r02/attempt1/run_20261017-004102` |
| fair_tuning_ext__c005_r00 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c005_r00/attempt1/run_20261017-004102` |
| fair_tuning_ext__c005_r01 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c005_r01/attempt1/run_20261017-004102` |
| fair_tuning_ext__c005_r02 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c005_r02/attempt1/run_20261017-004102` |
| fair_tuning_ext__c006_r00 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c006_r00/attempt1/run_20261017-004114` |
| fair_tuning_ext__c006_r01 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.7 | `runs/fair_tuning_ext__c006_r01/attempt1/run_20261017-004114` |
| fair_tuning_ext__c006_r02 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.7 | `runs/fair_tuning_ext__c006_r02/attempt1/run_20261017-004114` |
| fair_tuning_ext__c007_r00 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c007_r00/attempt1/run_20261017-004115` |
| fair_tuning_ext__c007_r01 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c007_r01/attempt1/run_20261017-004115` |
| fair_tuning_ext__c007_r02 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext__c007_r02/attempt1/run_20261017-004115` |
| fair_tuning_ext__c008_r00 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c008_r00/attempt1/run_20261017-004126` |
| fair_tuning_ext__c008_r01 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.7 | `runs/fair_tuning_ext__c008_r01/attempt1/run_20261017-004126` |
| fair_tuning_ext__c008_r02 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c008_r02/attempt1/run_20261017-004126` |
| fair_tuning_ext__c009_r00 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c009_r00/attempt1/run_20261017-004127` |
| fair_tuning_ext__c009_r01 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c009_r01/attempt1/run_20261017-004127` |
| fair_tuning_ext__c009_r02 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c009_r02/attempt1/run_20261017-004128` |
| fair_tuning_ext__c010_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c010_r00/attempt1/run_20261017-004138` |
| fair_tuning_ext__c010_r01 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c010_r01/attempt1/run_20261017-004138` |
| fair_tuning_ext__c010_r02 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c010_r02/attempt1/run_20261017-004138` |
| fair_tuning_ext__c011_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c011_r00/attempt1/run_20261017-004140` |
| fair_tuning_ext__c011_r01 | `{"--l2": 0.1, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c011_r01/attempt1/run_20261017-004140` |
| fair_tuning_ext__c011_r02 | `{"--l2": 0.1, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c011_r02/attempt1/run_20261017-004141` |
| fair_tuning_ext__c012_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c012_r00/attempt1/run_20261017-004150` |
| fair_tuning_ext__c012_r01 | `{"--l2": 1.0, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c012_r01/attempt1/run_20261017-004150` |
| fair_tuning_ext__c012_r02 | `{"--l2": 1.0, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c012_r02/attempt1/run_20261017-004150` |
| fair_tuning_ext__c013_r00 | `{"--l2": 10.0, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c013_r00/attempt1/run_20261017-004152` |
| fair_tuning_ext__c013_r01 | `{"--l2": 10.0, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c013_r01/attempt1/run_20261017-004152` |
| fair_tuning_ext__c013_r02 | `{"--l2": 10.0, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext__c013_r02/attempt1/run_20261017-004155` |
| fair_tuning_ext__c014_r00 | `{"--l2": 100.0, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 11.5 | `runs/fair_tuning_ext__c014_r00/attempt1/run_20261017-004203` |
| fair_tuning_ext__c014_r01 | `{"--l2": 100.0, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 11.5 | `runs/fair_tuning_ext__c014_r01/attempt1/run_20261017-004203` |
| fair_tuning_ext__c014_r02 | `{"--l2": 100.0, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.4 | `runs/fair_tuning_ext__c014_r02/attempt1/run_20261017-004203` |
| fair_tuning_ext__c015_r00 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext__c015_r00/attempt1/run_20261017-004205` |
| fair_tuning_ext__c015_r01 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c015_r01/attempt1/run_20261017-004205` |
| fair_tuning_ext__c015_r02 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.8 | `runs/fair_tuning_ext__c015_r02/attempt1/run_20261017-004208` |
| fair_tuning_ext__c016_r00 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c016_r00/attempt1/run_20261017-004214` |
| fair_tuning_ext__c016_r01 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c016_r01/attempt1/run_20261017-004214` |
| fair_tuning_ext__c016_r02 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c016_r02/attempt1/run_20261017-004214` |
| fair_tuning_ext__c017_r00 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext__c017_r00/attempt1/run_20261017-004218` |
| fair_tuning_ext__c017_r01 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext__c017_r01/attempt1/run_20261017-004218` |
| fair_tuning_ext__c017_r02 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.5 | `runs/fair_tuning_ext__c017_r02/attempt1/run_20261017-004222` |
| fair_tuning_ext__c018_r00 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c018_r00/attempt1/run_20261017-004227` |
| fair_tuning_ext__c018_r01 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c018_r01/attempt1/run_20261017-004227` |
| fair_tuning_ext__c018_r02 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c018_r02/attempt1/run_20261017-004227` |
| fair_tuning_ext__c019_r00 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c019_r00/attempt1/run_20261017-004231` |
| fair_tuning_ext__c019_r01 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.1 | `runs/fair_tuning_ext__c019_r01/attempt1/run_20261017-004231` |
| fair_tuning_ext__c019_r02 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.4 | `runs/fair_tuning_ext__c019_r02/attempt1/run_20261017-004235` |
| fair_tuning_ext__c020_r00 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c020_r00/attempt1/run_20261017-004239` |
| fair_tuning_ext__c020_r01 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c020_r01/attempt1/run_20261017-004239` |
| fair_tuning_ext__c020_r02 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c020_r02/attempt1/run_20261017-004239` |
| fair_tuning_ext__c021_r00 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c021_r00/attempt1/run_20261017-004244` |
| fair_tuning_ext__c021_r01 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c021_r01/attempt1/run_20261017-004244` |
| fair_tuning_ext__c021_r02 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c021_r02/attempt1/run_20261017-004248` |
| fair_tuning_ext__c022_r00 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.7 | `runs/fair_tuning_ext__c022_r00/attempt1/run_20261017-004251` |
| fair_tuning_ext__c022_r01 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c022_r01/attempt1/run_20261017-004251` |
| fair_tuning_ext__c022_r02 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.7 | `runs/fair_tuning_ext__c022_r02/attempt1/run_20261017-004251` |
| fair_tuning_ext__c023_r00 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c023_r00/attempt1/run_20261017-004257` |
| fair_tuning_ext__c023_r01 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c023_r01/attempt1/run_20261017-004257` |
| fair_tuning_ext__c023_r02 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c023_r02/attempt1/run_20261017-004301` |
| fair_tuning_ext__c024_r00 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c024_r00/attempt1/run_20261017-004302` |
| fair_tuning_ext__c024_r01 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c024_r01/attempt1/run_20261017-004302` |
| fair_tuning_ext__c024_r02 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c024_r02/attempt1/run_20261017-004302` |
| fair_tuning_ext__c025_r00 | `{"--l2": 0.01, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c025_r00/attempt1/run_20261017-004309` |
| fair_tuning_ext__c025_r01 | `{"--l2": 0.01, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c025_r01/attempt1/run_20261017-004309` |
| fair_tuning_ext__c025_r02 | `{"--l2": 0.01, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c025_r02/attempt1/run_20261017-004314` |
| fair_tuning_ext__c026_r00 | `{"--l2": 0.1, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c026_r00/attempt1/run_20261017-004314` |
| fair_tuning_ext__c026_r01 | `{"--l2": 0.1, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c026_r01/attempt1/run_20261017-004314` |
| fair_tuning_ext__c026_r02 | `{"--l2": 0.1, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c026_r02/attempt1/run_20261017-004314` |
| fair_tuning_ext__c027_r00 | `{"--l2": 1.0, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext__c027_r00/attempt1/run_20261017-004322` |
| fair_tuning_ext__c027_r01 | `{"--l2": 1.0, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext__c027_r01/attempt1/run_20261017-004322` |
| fair_tuning_ext__c027_r02 | `{"--l2": 1.0, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext__c027_r02/attempt1/run_20261017-004326` |
| fair_tuning_ext__c028_r00 | `{"--l2": 10.0, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c028_r00/attempt1/run_20261017-004326` |
| fair_tuning_ext__c028_r01 | `{"--l2": 10.0, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c028_r01/attempt1/run_20261017-004326` |
| fair_tuning_ext__c028_r02 | `{"--l2": 10.0, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c028_r02/attempt1/run_20261017-004326` |
| fair_tuning_ext__c029_r00 | `{"--l2": 100.0, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c029_r00/attempt1/run_20261017-004335` |
| fair_tuning_ext__c029_r01 | `{"--l2": 100.0, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c029_r01/attempt1/run_20261017-004335` |
| fair_tuning_ext__c029_r02 | `{"--l2": 100.0, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.6 | `runs/fair_tuning_ext__c029_r02/attempt1/run_20261017-004338` |
| fair_tuning_ext__c030_r00 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c030_r00/attempt1/run_20261017-004338` |
| fair_tuning_ext__c030_r01 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c030_r01/attempt1/run_20261017-004338` |
| fair_tuning_ext__c030_r02 | `{"--hidden": 7, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c030_r02/attempt1/run_20261017-004338` |
| fair_tuning_ext__c031_r00 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.2 | `runs/fair_tuning_ext__c031_r00/attempt1/run_20261017-004347` |
| fair_tuning_ext__c031_r01 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.3 | `runs/fair_tuning_ext__c031_r01/attempt1/run_20261017-004347` |
| fair_tuning_ext__c031_r02 | `{"--hidden": 7, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.3 | `runs/fair_tuning_ext__c031_r02/attempt1/run_20261017-004351` |
| fair_tuning_ext__c032_r00 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c032_r00/attempt1/run_20261017-004351` |
| fair_tuning_ext__c032_r01 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c032_r01/attempt1/run_20261017-004351` |
| fair_tuning_ext__c032_r02 | `{"--hidden": 7, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c032_r02/attempt1/run_20261017-004351` |
| fair_tuning_ext__c033_r00 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.3 | `runs/fair_tuning_ext__c033_r00/attempt1/run_20261017-004400` |
| fair_tuning_ext__c033_r01 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.1 | `runs/fair_tuning_ext__c033_r01/attempt1/run_20261017-004400` |
| fair_tuning_ext__c033_r02 | `{"--hidden": 7, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c033_r02/attempt1/run_20261017-004403` |
| fair_tuning_ext__c034_r00 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext__c034_r00/attempt1/run_20261017-004403` |
| fair_tuning_ext__c034_r01 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c034_r01/attempt1/run_20261017-004403` |
| fair_tuning_ext__c034_r02 | `{"--hidden": 7, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c034_r02/attempt1/run_20261017-004403` |
| fair_tuning_ext__c035_r00 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c035_r00/attempt1/run_20261017-004413` |
| fair_tuning_ext__c035_r01 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c035_r01/attempt1/run_20261017-004413` |
| fair_tuning_ext__c035_r02 | `{"--hidden": 9, "--l2": 0.01, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c035_r02/attempt1/run_20261017-004416` |
| fair_tuning_ext__c036_r00 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c036_r00/attempt1/run_20261017-004416` |
| fair_tuning_ext__c036_r01 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c036_r01/attempt1/run_20261017-004416` |
| fair_tuning_ext__c036_r02 | `{"--hidden": 9, "--l2": 0.1, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c036_r02/attempt1/run_20261017-004416` |
| fair_tuning_ext__c037_r00 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c037_r00/attempt1/run_20261017-004426` |
| fair_tuning_ext__c037_r01 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c037_r01/attempt1/run_20261017-004426` |
| fair_tuning_ext__c037_r02 | `{"--hidden": 9, "--l2": 1.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c037_r02/attempt1/run_20261017-004428` |
| fair_tuning_ext__c038_r00 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c038_r00/attempt1/run_20261017-004428` |
| fair_tuning_ext__c038_r01 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.6 | `runs/fair_tuning_ext__c038_r01/attempt1/run_20261017-004428` |
| fair_tuning_ext__c038_r02 | `{"--hidden": 9, "--l2": 10.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c038_r02/attempt1/run_20261017-004428` |
| fair_tuning_ext__c039_r00 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c039_r00/attempt1/run_20261017-004438` |
| fair_tuning_ext__c039_r01 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext__c039_r01/attempt1/run_20261017-004438` |
| fair_tuning_ext__c039_r02 | `{"--hidden": 9, "--l2": 100.0, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c039_r02/attempt1/run_20261017-004439` |
| fair_tuning_ext__c040_r00 | `{"--l2": 0.01, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 11.3 | `runs/fair_tuning_ext__c040_r00/attempt1/run_20261017-004439` |
| fair_tuning_ext__c040_r01 | `{"--l2": 0.01, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 11.5 | `runs/fair_tuning_ext__c040_r01/attempt1/run_20261017-004439` |
| fair_tuning_ext__c040_r02 | `{"--l2": 0.01, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext__c040_r02/attempt1/run_20261017-004441` |
| fair_tuning_ext__c041_r00 | `{"--l2": 0.1, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c041_r00/attempt1/run_20261017-004450` |
| fair_tuning_ext__c041_r01 | `{"--l2": 0.1, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c041_r01/attempt1/run_20261017-004450` |
| fair_tuning_ext__c041_r02 | `{"--l2": 0.1, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext__c041_r02/attempt1/run_20261017-004451` |
| fair_tuning_ext__c042_r00 | `{"--l2": 1.0, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext__c042_r00/attempt1/run_20261017-004451` |
| fair_tuning_ext__c042_r01 | `{"--l2": 1.0, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext__c042_r01/attempt1/run_20261017-004452` |
| fair_tuning_ext__c042_r02 | `{"--l2": 1.0, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext__c042_r02/attempt1/run_20261017-004454` |
| fair_tuning_ext__c043_r00 | `{"--l2": 10.0, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c043_r00/attempt1/run_20261017-004502` |
| fair_tuning_ext__c043_r01 | `{"--l2": 10.0, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c043_r01/attempt1/run_20261017-004502` |
| fair_tuning_ext__c043_r02 | `{"--l2": 10.0, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext__c043_r02/attempt1/run_20261017-004503` |
| fair_tuning_ext__c044_r00 | `{"--l2": 100.0, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext__c044_r00/attempt1/run_20261017-004503` |
| fair_tuning_ext__c044_r01 | `{"--l2": 100.0, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext__c044_r01/attempt1/run_20261017-004505` |
| fair_tuning_ext__c044_r02 | `{"--l2": 100.0, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext__c044_r02/attempt1/run_20261017-004508` |

#### `fair_tuning_ext2`

- Status COMPLETED; 2026-10-16T22:45:58Z to 2026-10-16T22:47:59Z; commit e7beb6a4f0 (dirty=True); paper_code_sha256 b1e65643b4fb (unchanged)
- Coverage: COMPLETE SWEEP. Planned 54, expected 54, completed 54, duplicates 0, failed 0, runs with a retry 0
- Summary: `summary.json`. Command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/simon/Documents/Reproduced papers/reproduced_papers/implementation.py --paper contextuality_inductive_bias --config configs/fair_quantum_tuning.json --outdir /Users/simon/Documents/Reproduced papers/reproduced_papers/papers/contextuality_inductive_bias/outdir/sweeps/fair_tuning_ext2/runs/fair_tuning_ext2__c000_r00/attempt1 --sweep-id fair_tuning_ext2 ...`

| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |
|---|---|---:|---:|---|---:|---|
| fair_tuning_ext2__c000_r00 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.7 | `runs/fair_tuning_ext2__c000_r00/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c000_r01 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.5 | `runs/fair_tuning_ext2__c000_r01/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c000_r02 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.7 | `runs/fair_tuning_ext2__c000_r02/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c001_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.6 | `runs/fair_tuning_ext2__c001_r00/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c001_r01 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 13.5 | `runs/fair_tuning_ext2__c001_r01/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c001_r02 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 13.7 | `runs/fair_tuning_ext2__c001_r02/attempt1/run_20261017-004602` |
| fair_tuning_ext2__c002_r00 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 13.1 | `runs/fair_tuning_ext2__c002_r00/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c002_r01 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c002_r01/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c002_r02 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext2__c002_r02/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c003_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext2__c003_r00/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c003_r01 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 13.1 | `runs/fair_tuning_ext2__c003_r01/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c003_r02 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c003_r02/attempt1/run_20261017-004615` |
| fair_tuning_ext2__c004_r00 | `{"--l2": 0.0001, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext2__c004_r00/attempt1/run_20261017-004628` |
| fair_tuning_ext2__c004_r01 | `{"--l2": 0.0001, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c004_r01/attempt1/run_20261017-004628` |
| fair_tuning_ext2__c004_r02 | `{"--l2": 0.0001, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext2__c004_r02/attempt1/run_20261017-004628` |
| fair_tuning_ext2__c005_r00 | `{"--l2": 0.001, "--lr": 0.001, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext2__c005_r00/attempt1/run_20261017-004628` |
| fair_tuning_ext2__c005_r01 | `{"--l2": 0.001, "--lr": 0.001, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext2__c005_r01/attempt1/run_20261017-004628` |
| fair_tuning_ext2__c005_r02 | `{"--l2": 0.001, "--lr": 0.001, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 14.1 | `runs/fair_tuning_ext2__c005_r02/attempt1/run_20261017-004630` |
| fair_tuning_ext2__c006_r00 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext2__c006_r00/attempt1/run_20261017-004641` |
| fair_tuning_ext2__c006_r01 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.9 | `runs/fair_tuning_ext2__c006_r01/attempt1/run_20261017-004641` |
| fair_tuning_ext2__c006_r02 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c006_r02/attempt1/run_20261017-004641` |
| fair_tuning_ext2__c007_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c007_r00/attempt1/run_20261017-004641` |
| fair_tuning_ext2__c007_r01 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.6 | `runs/fair_tuning_ext2__c007_r01/attempt1/run_20261017-004641` |
| fair_tuning_ext2__c007_r02 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 14.0 | `runs/fair_tuning_ext2__c007_r02/attempt1/run_20261017-004644` |
| fair_tuning_ext2__c008_r00 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext2__c008_r00/attempt1/run_20261017-004654` |
| fair_tuning_ext2__c008_r01 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext2__c008_r01/attempt1/run_20261017-004654` |
| fair_tuning_ext2__c008_r02 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext2__c008_r02/attempt1/run_20261017-004654` |
| fair_tuning_ext2__c009_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext2__c009_r00/attempt1/run_20261017-004654` |
| fair_tuning_ext2__c009_r01 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 11.8 | `runs/fair_tuning_ext2__c009_r01/attempt1/run_20261017-004654` |
| fair_tuning_ext2__c009_r02 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.003, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 13.8 | `runs/fair_tuning_ext2__c009_r02/attempt1/run_20261017-004659` |
| fair_tuning_ext2__c010_r00 | `{"--l2": 0.0001, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext2__c010_r00/attempt1/run_20261017-004706` |
| fair_tuning_ext2__c010_r01 | `{"--l2": 0.0001, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext2__c010_r01/attempt1/run_20261017-004706` |
| fair_tuning_ext2__c010_r02 | `{"--l2": 0.0001, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext2__c010_r02/attempt1/run_20261017-004706` |
| fair_tuning_ext2__c011_r00 | `{"--l2": 0.001, "--lr": 0.003, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 11.9 | `runs/fair_tuning_ext2__c011_r00/attempt1/run_20261017-004706` |
| fair_tuning_ext2__c011_r01 | `{"--l2": 0.001, "--lr": 0.003, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext2__c011_r01/attempt1/run_20261017-004706` |
| fair_tuning_ext2__c011_r02 | `{"--l2": 0.001, "--lr": 0.003, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 14.4 | `runs/fair_tuning_ext2__c011_r02/attempt1/run_20261017-004713` |
| fair_tuning_ext2__c012_r00 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 13.0 | `runs/fair_tuning_ext2__c012_r00/attempt1/run_20261017-004718` |
| fair_tuning_ext2__c012_r01 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext2__c012_r01/attempt1/run_20261017-004718` |
| fair_tuning_ext2__c012_r02 | `{"--hidden": 7, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 12.8 | `runs/fair_tuning_ext2__c012_r02/attempt1/run_20261017-004718` |
| fair_tuning_ext2__c013_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_biased"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext2__c013_r00/attempt1/run_20261017-004718` |
| fair_tuning_ext2__c013_r01 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_biased"}` | 1 | 1 | COMPLETED | 12.7 | `runs/fair_tuning_ext2__c013_r01/attempt1/run_20261017-004718` |
| fair_tuning_ext2__c013_r02 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_biased"}` | 2 | 1 | COMPLETED | 14.2 | `runs/fair_tuning_ext2__c013_r02/attempt1/run_20261017-004727` |
| fair_tuning_ext2__c014_r00 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.5 | `runs/fair_tuning_ext2__c014_r00/attempt1/run_20261017-004731` |
| fair_tuning_ext2__c014_r01 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext2__c014_r01/attempt1/run_20261017-004731` |
| fair_tuning_ext2__c014_r02 | `{"--hidden": 9, "--l2": 0.0001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 12.2 | `runs/fair_tuning_ext2__c014_r02/attempt1/run_20261017-004731` |
| fair_tuning_ext2__c015_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 0 | 1 | COMPLETED | 12.1 | `runs/fair_tuning_ext2__c015_r00/attempt1/run_20261017-004731` |
| fair_tuning_ext2__c015_r01 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 1 | 1 | COMPLETED | 12.0 | `runs/fair_tuning_ext2__c015_r01/attempt1/run_20261017-004731` |
| fair_tuning_ext2__c015_r02 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.01, "--model": "mlp_unbiased"}` | 2 | 1 | COMPLETED | 14.2 | `runs/fair_tuning_ext2__c015_r02/attempt1/run_20261017-004741` |
| fair_tuning_ext2__c016_r00 | `{"--l2": 0.0001, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext2__c016_r00/attempt1/run_20261017-004743` |
| fair_tuning_ext2__c016_r01 | `{"--l2": 0.0001, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext2__c016_r01/attempt1/run_20261017-004743` |
| fair_tuning_ext2__c016_r02 | `{"--l2": 0.0001, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext2__c016_r02/attempt1/run_20261017-004743` |
| fair_tuning_ext2__c017_r00 | `{"--l2": 0.001, "--lr": 0.01, "--model": "quadratic"}` | 0 | 1 | COMPLETED | 12.4 | `runs/fair_tuning_ext2__c017_r00/attempt1/run_20261017-004743` |
| fair_tuning_ext2__c017_r01 | `{"--l2": 0.001, "--lr": 0.01, "--model": "quadratic"}` | 1 | 1 | COMPLETED | 12.3 | `runs/fair_tuning_ext2__c017_r01/attempt1/run_20261017-004743` |
| fair_tuning_ext2__c017_r02 | `{"--l2": 0.001, "--lr": 0.01, "--model": "quadratic"}` | 2 | 1 | COMPLETED | 9.0 | `runs/fair_tuning_ext2__c017_r02/attempt1/run_20261017-004754` |

#### `fair_final`

- Status PARTIAL; 2026-10-16T22:48:28Z to 2026-10-16T23:50:09Z; commit e7beb6a4f0 (dirty=True); paper_code_sha256 556da8fd778b (unchanged)
- Coverage: PARTIAL SWEEP. Planned 100, expected 100, completed 99, duplicates 0, failed 1, runs with a retry 1
- Summary: `summary.json`. Command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/simon/Documents/Reproduced papers/reproduced_papers/implementation.py --paper contextuality_inductive_bias --config configs/fair_quantum_tuning.json --outdir /Users/simon/Documents/Reproduced papers/reproduced_papers/papers/contextuality_inductive_bias/outdir/sweeps/fair_final/runs/fair_final__c000_r00/attempt1 --sweep-id fair_final ...`

| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |
|---|---|---:|---:|---|---:|---|
| fair_final__c000_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 10 | 1 | COMPLETED | 151.5 | `runs/fair_final__c000_r00/attempt1/run_20261017-004831` |
| fair_final__c000_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 11 | 1 | COMPLETED | 149.6 | `runs/fair_final__c000_r01/attempt1/run_20261017-004831` |
| fair_final__c000_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 12 | 1 | COMPLETED | 149.5 | `runs/fair_final__c000_r02/attempt1/run_20261017-004831` |
| fair_final__c000_r03 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 13 | 1 | COMPLETED | 149.0 | `runs/fair_final__c000_r03/attempt1/run_20261017-004831` |
| fair_final__c000_r04 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 14 | 1 | COMPLETED | 149.5 | `runs/fair_final__c000_r04/attempt1/run_20261017-004831` |
| fair_final__c000_r05 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 15 | 1 | COMPLETED | 150.2 | `runs/fair_final__c000_r05/attempt1/run_20261017-004831` |
| fair_final__c000_r06 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 16 | 1 | COMPLETED | 146.8 | `runs/fair_final__c000_r06/attempt1/run_20261017-005101` |
| fair_final__c000_r07 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 17 | 1 | COMPLETED | 148.8 | `runs/fair_final__c000_r07/attempt1/run_20261017-005101` |
| fair_final__c000_r08 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 18 | 1 | COMPLETED | 146.8 | `runs/fair_final__c000_r08/attempt1/run_20261017-005101` |
| fair_final__c000_r09 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum"}` | 19 | 1 | COMPLETED | 146.3 | `runs/fair_final__c000_r09/attempt1/run_20261017-005101` |
| fair_final__c001_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 10 | 1 | COMPLETED | 236.8 | `runs/fair_final__c001_r00/attempt1/run_20261017-005102` |
| fair_final__c001_r01 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 11 | 1 | COMPLETED | 239.8 | `runs/fair_final__c001_r01/attempt1/run_20261017-005104` |
| fair_final__c001_r02 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 12 | 1 | COMPLETED | 236.2 | `runs/fair_final__c001_r02/attempt1/run_20261017-005328` |
| fair_final__c001_r03 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 13 | 1 | COMPLETED | 238.7 | `runs/fair_final__c001_r03/attempt1/run_20261017-005328` |
| fair_final__c001_r04 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 14 | 1 | COMPLETED | 235.9 | `runs/fair_final__c001_r04/attempt1/run_20261017-005328` |
| fair_final__c001_r05 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 15 | 1 | COMPLETED | 236.5 | `runs/fair_final__c001_r05/attempt1/run_20261017-005331` |
| fair_final__c001_r06 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 16 | 1 | COMPLETED | 242.6 | `runs/fair_final__c001_r06/attempt1/run_20261017-005501` |
| fair_final__c001_r07 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 17 | 1 | COMPLETED | 242.7 | `runs/fair_final__c001_r07/attempt1/run_20261017-005505` |
| fair_final__c001_r08 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 18 | 1 | COMPLETED | 259.7 | `runs/fair_final__c001_r08/attempt1/run_20261017-005724` |
| fair_final__c001_r09 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum"}` | 19 | 1 | COMPLETED | 259.2 | `runs/fair_final__c001_r09/attempt1/run_20261017-005724` |
| fair_final__c002_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 10 | 1 | COMPLETED | 564.2 | `runs/fair_final__c002_r00/attempt1/run_20261017-005727` |
| fair_final__c002_r01 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 11 | 1 | COMPLETED | 562.9 | `runs/fair_final__c002_r01/attempt1/run_20261017-005727` |
| fair_final__c002_r02 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 12 | 1 | COMPLETED | 585.1 | `runs/fair_final__c002_r02/attempt1/run_20261017-005904` |
| fair_final__c002_r03 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 13 | 1 | COMPLETED | 587.6 | `runs/fair_final__c002_r03/attempt1/run_20261017-005908` |
| fair_final__c002_r04 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 14 | 1 | COMPLETED | 598.7 | `runs/fair_final__c002_r04/attempt1/run_20261017-010145` |
| fair_final__c002_r05 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 15 | 1 | COMPLETED | 599.3 | `runs/fair_final__c002_r05/attempt1/run_20261017-010145` |
| fair_final__c002_r06 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 16 | 1 | COMPLETED | 535.2 | `runs/fair_final__c002_r06/attempt1/run_20261017-010651` |
| fair_final__c002_r07 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 17 | 1 | COMPLETED | 536.4 | `runs/fair_final__c002_r07/attempt1/run_20261017-010652` |
| fair_final__c002_r08 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 18 | 1 | COMPLETED | 508.1 | `runs/fair_final__c002_r08/attempt1/run_20261017-010851` |
| fair_final__c002_r09 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group"}` | 19 | 1 | COMPLETED | 508.8 | `runs/fair_final__c002_r09/attempt1/run_20261017-010858` |
| fair_final__c003_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 10 | 1 | COMPLETED | 16.6 | `runs/fair_final__c003_r00/attempt1/run_20261017-011144` |
| fair_final__c003_r01 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 11 | 1 | COMPLETED | 17.0 | `runs/fair_final__c003_r01/attempt1/run_20261017-011145` |
| fair_final__c003_r02 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 12 | 1 | COMPLETED | 17.1 | `runs/fair_final__c003_r02/attempt1/run_20261017-011201` |
| fair_final__c003_r03 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 13 | 1 | COMPLETED | 17.3 | `runs/fair_final__c003_r03/attempt1/run_20261017-011202` |
| fair_final__c003_r04 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 14 | 1 | COMPLETED | 17.1 | `runs/fair_final__c003_r04/attempt1/run_20261017-011218` |
| fair_final__c003_r05 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 15 | 1 | COMPLETED | 17.1 | `runs/fair_final__c003_r05/attempt1/run_20261017-011219` |
| fair_final__c003_r06 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 16 | 1 | COMPLETED | 17.1 | `runs/fair_final__c003_r06/attempt1/run_20261017-011235` |
| fair_final__c003_r07 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 17 | 1 | COMPLETED | 17.1 | `runs/fair_final__c003_r07/attempt1/run_20261017-011237` |
| fair_final__c003_r08 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 18 | 1 | COMPLETED | 17.3 | `runs/fair_final__c003_r08/attempt1/run_20261017-011252` |
| fair_final__c003_r09 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased"}` | 19 | 1 | COMPLETED | 17.2 | `runs/fair_final__c003_r09/attempt1/run_20261017-011254` |
| fair_final__c004_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 10 | 1 | COMPLETED | 16.7 | `runs/fair_final__c004_r00/attempt1/run_20261017-011310` |
| fair_final__c004_r01 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 11 | 1 | COMPLETED | 16.6 | `runs/fair_final__c004_r01/attempt1/run_20261017-011311` |
| fair_final__c004_r02 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 12 | 1 | COMPLETED | 16.5 | `runs/fair_final__c004_r02/attempt1/run_20261017-011326` |
| fair_final__c004_r03 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 13 | 1 | COMPLETED | 16.6 | `runs/fair_final__c004_r03/attempt1/run_20261017-011328` |
| fair_final__c004_r04 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 14 | 1 | COMPLETED | 16.6 | `runs/fair_final__c004_r04/attempt1/run_20261017-011343` |
| fair_final__c004_r05 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 15 | 1 | COMPLETED | 16.7 | `runs/fair_final__c004_r05/attempt1/run_20261017-011344` |
| fair_final__c004_r06 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 16 | 1 | COMPLETED | 16.7 | `runs/fair_final__c004_r06/attempt1/run_20261017-011400` |
| fair_final__c004_r07 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 17 | 1 | COMPLETED | 16.7 | `runs/fair_final__c004_r07/attempt1/run_20261017-011401` |
| fair_final__c004_r08 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 18 | 1 | COMPLETED | 16.8 | `runs/fair_final__c004_r08/attempt1/run_20261017-011416` |
| fair_final__c004_r09 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased"}` | 19 | 1 | COMPLETED | 16.7 | `runs/fair_final__c004_r09/attempt1/run_20261017-011418` |
| fair_final__c005_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 10 | 1 | COMPLETED | 170.8 | `runs/fair_final__c005_r00/attempt1/run_20261017-011433` |
| fair_final__c005_r01 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 11 | 1 | COMPLETED | 171.1 | `runs/fair_final__c005_r01/attempt1/run_20261017-011435` |
| fair_final__c005_r02 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 12 | 1 | COMPLETED | 181.2 | `runs/fair_final__c005_r02/attempt1/run_20261017-011605` |
| fair_final__c005_r03 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 13 | 1 | COMPLETED | 180.1 | `runs/fair_final__c005_r03/attempt1/run_20261017-011605` |
| fair_final__c005_r04 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 14 | 1 | COMPLETED | 159.5 | `runs/fair_final__c005_r04/attempt1/run_20261017-011717` |
| fair_final__c005_r05 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 15 | 1 | COMPLETED | 157.2 | `runs/fair_final__c005_r05/attempt1/run_20261017-011722` |
| fair_final__c005_r06 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 16 | 1 | COMPLETED | 157.7 | `runs/fair_final__c005_r06/attempt1/run_20261017-011722` |
| fair_final__c005_r07 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 17 | 1 | COMPLETED | 156.7 | `runs/fair_final__c005_r07/attempt1/run_20261017-011724` |
| fair_final__c005_r08 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 18 | 1 | COMPLETED | 157.4 | `runs/fair_final__c005_r08/attempt1/run_20261017-011848` |
| fair_final__c005_r09 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum"}` | 19 | 1 | COMPLETED | 158.1 | `runs/fair_final__c005_r09/attempt1/run_20261017-011849` |
| fair_final__c006_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 10 | 1 | COMPLETED | 241.4 | `runs/fair_final__c006_r00/attempt1/run_20261017-011956` |
| fair_final__c006_r01 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 11 | 1 | COMPLETED | 243.7 | `runs/fair_final__c006_r01/attempt1/run_20261017-011959` |
| fair_final__c006_r02 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 12 | 1 | COMPLETED | 242.2 | `runs/fair_final__c006_r02/attempt1/run_20261017-012000` |
| fair_final__c006_r03 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 13 | 1 | COMPLETED | 239.9 | `runs/fair_final__c006_r03/attempt1/run_20261017-012000` |
| fair_final__c006_r04 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 14 | 1 | COMPLETED | 245.4 | `runs/fair_final__c006_r04/attempt1/run_20261017-012125` |
| fair_final__c006_r05 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 15 | 1 | COMPLETED | 246.0 | `runs/fair_final__c006_r05/attempt1/run_20261017-012127` |
| fair_final__c006_r06 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 16 | 1 | COMPLETED | 255.4 | `runs/fair_final__c006_r06/attempt1/run_20261017-012358` |
| fair_final__c006_r07 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 17 | 1 | COMPLETED | 257.8 | `runs/fair_final__c006_r07/attempt1/run_20261017-012401` |
| fair_final__c006_r08 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 18 | 1 | COMPLETED | 259.5 | `runs/fair_final__c006_r08/attempt1/run_20261017-012403` |
| fair_final__c006_r09 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum"}` | 19 | 1 | COMPLETED | 259.2 | `runs/fair_final__c006_r09/attempt1/run_20261017-012403` |
| fair_final__c007_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 10 | 1 | COMPLETED | 517.2 | `runs/fair_final__c007_r00/attempt1/run_20261017-012531` |
| fair_final__c007_r01 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 11 | 1 | COMPLETED | 524.4 | `runs/fair_final__c007_r01/attempt1/run_20261017-012533` |
| fair_final__c007_r02 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 12 | 1 | COMPLETED | 549.6 | `runs/fair_final__c007_r02/attempt1/run_20261017-012814` |
| fair_final__c007_r03 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 13 | 1 | COMPLETED | 540.4 | `runs/fair_final__c007_r03/attempt1/run_20261017-012820` |
| fair_final__c007_r04 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 14 | 1 | COMPLETED | 540.3 | `runs/fair_final__c007_r04/attempt1/run_20261017-012823` |
| fair_final__c007_r05 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 15 | 1 | COMPLETED | 540.1 | `runs/fair_final__c007_r05/attempt1/run_20261017-012823` |
| fair_final__c007_r06 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 16 | 1 | COMPLETED | 570.5 | `runs/fair_final__c007_r06/attempt1/run_20261017-013433` |
| fair_final__c007_r07 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 17 | 1 | COMPLETED | 546.9 | `runs/fair_final__c007_r07/attempt1/run_20261017-013433` |
| fair_final__c007_r08 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 18 | 1 | COMPLETED | 539.3 | `runs/fair_final__c007_r08/attempt1/run_20261017-013720` |
| fair_final__c007_r09 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1}` | 19 | 1 | COMPLETED | 529.2 | `runs/fair_final__c007_r09/attempt1/run_20261017-013722` |
| fair_final__c008_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 10 | 1 | COMPLETED | 14.4 | `runs/fair_final__c008_r00/attempt1/run_20261017-013722` |
| fair_final__c008_r01 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 11 | 1 | COMPLETED | 14.3 | `runs/fair_final__c008_r01/attempt1/run_20261017-013722` |
| fair_final__c008_r02 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 12 | 1 | COMPLETED | 15.8 | `runs/fair_final__c008_r02/attempt1/run_20261017-013737` |
| fair_final__c008_r03 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 13 | 1 | COMPLETED | 15.7 | `runs/fair_final__c008_r03/attempt1/run_20261017-013737` |
| fair_final__c008_r04 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 14 | 1 | COMPLETED | 15.7 | `runs/fair_final__c008_r04/attempt1/run_20261017-013753` |
| fair_final__c008_r05 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 15 | 1 | COMPLETED | 15.7 | `runs/fair_final__c008_r05/attempt1/run_20261017-013753` |
| fair_final__c008_r06 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 16 | 1 | COMPLETED | 15.8 | `runs/fair_final__c008_r06/attempt1/run_20261017-013809` |
| fair_final__c008_r07 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 17 | 1 | COMPLETED | 15.7 | `runs/fair_final__c008_r07/attempt1/run_20261017-013809` |
| fair_final__c008_r08 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 18 | 1 | COMPLETED | 15.9 | `runs/fair_final__c008_r08/attempt1/run_20261017-013825` |
| fair_final__c008_r09 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic"}` | 19 | 1 | COMPLETED | 15.7 | `runs/fair_final__c008_r09/attempt1/run_20261017-013825` |
| fair_final__c009_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 10 | 1 | COMPLETED | 263.4 | `runs/fair_final__c009_r00/attempt1/run_20261017-013841` |
| fair_final__c009_r01 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 11 | 2 | FAILED | 9.2 | `None` |
| fair_final__c009_r02 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 12 | 1 | COMPLETED | 275.0 | `runs/fair_final__c009_r02/attempt1/run_20261017-013859` |
| fair_final__c009_r03 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 13 | 1 | COMPLETED | 260.5 | `runs/fair_final__c009_r03/attempt1/run_20261017-014306` |
| fair_final__c009_r04 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 14 | 1 | COMPLETED | 260.6 | `runs/fair_final__c009_r04/attempt1/run_20261017-014326` |
| fair_final__c009_r05 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 15 | 1 | COMPLETED | 259.1 | `runs/fair_final__c009_r05/attempt1/run_20261017-014334` |
| fair_final__c009_r06 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 16 | 1 | COMPLETED | 266.0 | `runs/fair_final__c009_r06/attempt1/run_20261017-014339` |
| fair_final__c009_r07 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 17 | 1 | COMPLETED | 201.4 | `runs/fair_final__c009_r07/attempt1/run_20261017-014612` |
| fair_final__c009_r08 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 18 | 1 | COMPLETED | 199.1 | `runs/fair_final__c009_r08/attempt1/run_20261017-014621` |
| fair_final__c009_r09 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1}` | 19 | 1 | COMPLETED | 168.9 | `runs/fair_final__c009_r09/attempt1/run_20261017-014726` |

#### `fair_robustness`

- Status COMPLETED; 2026-10-16T23:50:54Z to 2026-10-17T00:09:14Z; commit e7beb6a4f0 (dirty=True); paper_code_sha256 556da8fd778b (unchanged)
- Coverage: COMPLETE SWEEP. Planned 30, expected 30, completed 30, duplicates 0, failed 0, runs with a retry 0
- Summary: `summary.json`. Command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/simon/Documents/Reproduced papers/reproduced_papers/implementation.py --paper contextuality_inductive_bias --config configs/fair_quantum_tuning.json --outdir /Users/simon/Documents/Reproduced papers/reproduced_papers/papers/contextuality_inductive_bias/outdir/sweeps/fair_robustness/runs/fair_robustness__c000_r00/attempt1 --sweep-id fair_robustness ...`

| Run ID | Candidate | Seed | Attempts | Status | Elapsed (s) | Run dir |
|---|---|---:|---:|---|---:|---|
| fair_robustness__c000_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 145.5 | `runs/fair_robustness__c000_r00/attempt1/run_20261017-015057` |
| fair_robustness__c001_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 240.7 | `runs/fair_robustness__c001_r00/attempt1/run_20261017-015057` |
| fair_robustness__c002_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 454.2 | `runs/fair_robustness__c002_r00/attempt1/run_20261017-015057` |
| fair_robustness__c003_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 13.0 | `runs/fair_robustness__c003_r00/attempt1/run_20261017-015057` |
| fair_robustness__c004_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 12.7 | `runs/fair_robustness__c004_r00/attempt1/run_20261017-015057` |
| fair_robustness__c005_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 164.5 | `runs/fair_robustness__c005_r00/attempt1/run_20261017-015057` |
| fair_robustness__c006_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 260.0 | `runs/fair_robustness__c006_r00/attempt1/run_20261017-015111` |
| fair_robustness__c007_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1, "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 459.7 | `runs/fair_robustness__c007_r00/attempt1/run_20261017-015111` |
| fair_robustness__c008_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic", "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 15.5 | `runs/fair_robustness__c008_r00/attempt1/run_20261017-015325` |
| fair_robustness__c009_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1, "--train-seed": 101, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 232.9 | `runs/fair_robustness__c009_r00/attempt1/run_20261017-015340` |
| fair_robustness__c010_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 157.6 | `runs/fair_robustness__c010_r00/attempt1/run_20261017-015344` |
| fair_robustness__c011_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 250.6 | `runs/fair_robustness__c011_r00/attempt1/run_20261017-015500` |
| fair_robustness__c012_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 462.5 | `runs/fair_robustness__c012_r00/attempt1/run_20261017-015532` |
| fair_robustness__c013_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 17.3 | `runs/fair_robustness__c013_r00/attempt1/run_20261017-015622` |
| fair_robustness__c014_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 16.7 | `runs/fair_robustness__c014_r00/attempt1/run_20261017-015639` |
| fair_robustness__c015_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 169.4 | `runs/fair_robustness__c015_r00/attempt1/run_20261017-015656` |
| fair_robustness__c016_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 261.0 | `runs/fair_robustness__c016_r00/attempt1/run_20261017-015733` |
| fair_robustness__c017_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1, "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 463.8 | `runs/fair_robustness__c017_r00/attempt1/run_20261017-015833` |
| fair_robustness__c018_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic", "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 15.5 | `runs/fair_robustness__c018_r00/attempt1/run_20261017-015852` |
| fair_robustness__c019_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1, "--train-seed": 102, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 229.4 | `runs/fair_robustness__c019_r00/attempt1/run_20261017-015907` |
| fair_robustness__c020_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 1, "--lr": 0.003, "--model": "generic_quantum", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 157.8 | `runs/fair_robustness__c020_r00/attempt1/run_20261017-015910` |
| fair_robustness__c021_r00 | `{"--cnot-pattern": "paper", "--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "generic_quantum", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 254.4 | `runs/fair_robustness__c021_r00/attempt1/run_20261017-015945` |
| fair_robustness__c022_r00 | `{"--computation-space": "fock", "--l2": 0.0, "--lr": 0.003, "--model": "photonic_merlin", "--n-photons": 2, "--readout": "group", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 384.1 | `runs/fair_robustness__c022_r00/attempt1/run_20261017-020148` |
| fair_robustness__c023_r00 | `{"--hidden": 7, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_biased", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 17.0 | `runs/fair_robustness__c023_r00/attempt1/run_20261017-020154` |
| fair_robustness__c024_r00 | `{"--hidden": 9, "--l2": 0.001, "--lr": 0.001, "--model": "mlp_unbiased", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 16.7 | `runs/fair_robustness__c024_r00/attempt1/run_20261017-020212` |
| fair_robustness__c025_r00 | `{"--l2": 0.0, "--layers": 1, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 175.1 | `runs/fair_robustness__c025_r00/attempt1/run_20261017-020228` |
| fair_robustness__c026_r00 | `{"--l2": 0.0, "--layers": 2, "--lr": 0.001, "--model": "biased_quantum", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 249.7 | `runs/fair_robustness__c026_r00/attempt1/run_20261017-020257` |
| fair_robustness__c027_r00 | `{"--l2": 0.1, "--lr": 0.001, "--model": "bias_fourier", "--spectrum": 1, "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 365.6 | `runs/fair_robustness__c027_r00/attempt1/run_20261017-020314` |
| fair_robustness__c028_r00 | `{"--l2": 0.01, "--lr": 0.001, "--model": "quadratic", "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 15.8 | `runs/fair_robustness__c028_r00/attempt1/run_20261017-020359` |
| fair_robustness__c029_r00 | `{"--l2": 1.0, "--lr": 0.001, "--model": "surrogate", "--spectrum": 1, "--train-seed": 103, "--train-source": "generated"}` | 10 | 1 | COMPLETED | 205.8 | `runs/fair_robustness__c029_r00/attempt1/run_20261017-020415` |

