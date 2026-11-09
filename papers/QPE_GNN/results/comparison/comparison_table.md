| Claim | System | Role | Provenance | Scope | Runs | Metric | Matching basis | Params | Wall-clock | Status |
|---|---|---|---|---|---:|---:|---|---:|---:|---|
| C1 | GRIT RRWP | paper baseline | paper Table 1 (reported) | paper (2000 epochs) | — | test_mae 0.0600 ± 0.0020 | paper protocol | — | — | reference only; budget differs from local runs |
| C1 | GRIT RRWP+1-CQRW | paper method | paper Table 1 (reported) | paper (2000 epochs) | — | test_mae 0.0580 ± 0.0020 | paper protocol | — | — | reference only; budget differs from local runs |
| C1 | GRIT RRWP+2-QiRW | paper method | paper Table 1 (reported) | paper (2000 epochs) | — | test_mae 0.0590 ± 0.0040 | paper protocol | — | — | reference only; budget differs from local runs |
| C1 | zinc_rrwp | paper baseline (reproduced) | local, reproduced-papers-1 | partial (200 of 2000 epochs, 4 seeds) | 4/4 | test_mae 0.0905 ± 0.0098 | identical GRIT, data, schedule, 200 epochs, seeds 0-3 | 473,473 | 17328 s (mean per run) | COMMENSURABLE |
| C1 | zinc_rrwp41 | width control (Phase 1 plan) | local, reproduced-papers-1 | partial (200 of 2000 epochs, 4 seeds) | 4/4 | test_mae 0.1040 ± 0.0028 | identical GRIT, data, schedule, 200 epochs, seeds 0-3; PE width matched (41 channels) | 476,033 | 17317 s (mean per run) | COMMENSURABLE |
| C1 | zinc_cqrw1 | paper method (reproduced) | local, reproduced-papers-1 | partial (200 of 2000 epochs, 4 seeds) | 4/4 | test_mae 0.0866 ± 0.0050 | identical GRIT, data, schedule, 200 epochs, seeds 0-3 | 476,033 | 17335 s (mean per run) | COMMENSURABLE |
| C1 | zinc_qirw2 | paper method (reproduced) | local, reproduced-papers-1 | partial (200 of 2000 epochs, 4 seeds) | 4/4 | test_mae 0.0870 ± 0.0052 | identical GRIT, data, schedule, 200 epochs, seeds 0-3 | 476,033 | 17355 s (mean per run) | COMMENSURABLE |
| P2 | zinc_photon2 | MerLin adaptation (simulated) | local, merlin-ai (CUDA 12.6) | partial (200 of 2000 epochs, 4 seeds), cross-machine | 4/4 | test_mae 0.0842 ± 0.0046 | identical GRIT, data, schedule, 200 epochs, seeds 0-3; other machine and code state | 476,033 | 18393 s (mean per run) | PARTIALLY_COMMENSURABLE (cross-machine, user override 2026-11-09) |
| C2 | two_particle_xy | paper method (reproduced) | local CPU | paper-accurate | 1/1 | distinguished_pairs 150/150 | same 25 graphs, 5 permutations, tolerance 1e-9 | — | 132 s (whole run (all methods)) | COMMENSURABLE |
| C2 | rrwp | paper baseline (reproduced) | local CPU | paper-accurate | 1/1 | distinguished_pairs 0/150 | same 25 graphs, 5 permutations, tolerance 1e-9 | — | 132 s (whole run (all methods)) | COMMENSURABLE |
| C2 | ising_p2_linked | paper method, labelled proxy (parameters unpublished) | local CPU | paper-accurate | 1/1 | distinguished_pairs 143/150 | same 25 graphs, 5 permutations, tolerance 1e-9 | — | 132 s (whole run (all methods)) | PARTIALLY_COMMENSURABLE (proxy, not the paper's circuit) |
| P1 | photon2_threshold | MerLin adaptation (simulated) | local CPU | paper protocol, simulated ideal photons | 1/1 | distinguished_pairs 143/150 | same graphs, permutations, tolerance as C2 | — | 20 s (this method, both families) | COMMENSURABLE |
| P1 | photon3_threshold | MerLin adaptation (simulated) | local CPU | paper protocol, simulated ideal photons | 1/1 | distinguished_pairs 150/150 | same graphs, permutations, tolerance as C2 | — | 754 s (this method, both families) | COMMENSURABLE |
| C3 | synthetic_quantum | paper method (reproduced) | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 1.0000 ± 0.0000 | same 800 graphs and split, 200 epochs, constant lr 1e-3 | 3,906 | 138 s (mean per run) | COMMENSURABLE |
| C3 | synthetic_laplacian | paper baseline (reproduced, iso-parameter) | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 0.4750 ± 0.0000 | same 800 graphs and split, 200 epochs, constant lr 1e-3; 3,906 params | 3,906 | 135 s (mean per run) | COMMENSURABLE |
| C3 | synthetic_rrwp | paper baseline (reproduced) | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 0.8687 ± 0.2625 | same 800 graphs and split, 200 epochs, constant lr 1e-3; GatedGCN 7,074 params | 7,074 | 129 s (mean per run) | COMMENSURABLE |
| C3 | synthetic_gcn | no-encoding control | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 0.4750 ± 0.0000 | same 800 graphs and split, 200 epochs, constant lr 1e-3; 3,298 params | 3,298 | 133 s (mean per run) | COMMENSURABLE |
| C3 | synthetic_degree | additional fair baseline (Phase 5, degree confounder) | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 1.0000 ± 0.0000 | same 800 graphs and split, 200 epochs, constant lr 1e-3; 3,906 params, same model as quantum | 3,906 | 143 s (mean per run) | COMMENSURABLE (newer code state, additive branch) |
| C3 | synthetic_rrwp_cosine | Phase 3 diagnostic (cosine lr decay) | local, reproduced-papers-1 | paper protocol, 4 seeds | 4/4 | test_acc 0.4937 ± 0.0375 | as synthetic_rrwp but cosine lr decay to 1e-6 | 7,074 | 132 s (mean per run) | diagnostic only: different schedule from the frozen protocol |

| Claim | A | B | Tie-rule verdict | p | Commensurability | Note |
|---|---|---|---|---:|---|---|
| C1 | zinc_rrwp | zinc_cqrw1 | not distinguishable | 0.0139 | COMMENSURABLE |  |
| C1 | zinc_cqrw1 | zinc_rrwp41 | zinc_cqrw1 better | 0.0029 | COMMENSURABLE |  |
| C1 | zinc_rrwp | zinc_qirw2 | not distinguishable | 0.435 | COMMENSURABLE |  |
| C1 | zinc_qirw2 | zinc_rrwp41 | not distinguishable | 0.0636 | COMMENSURABLE |  |
| P2 | zinc_rrwp | zinc_photon2 | not distinguishable | 0.0726 | PARTIALLY_COMMENSURABLE | cross-machine; user override 2026-11-09 |
| P2 | zinc_rrwp41 | zinc_photon2 | zinc_photon2 better | 0.00507 | PARTIALLY_COMMENSURABLE | cross-machine; user override 2026-11-09 |
| P2 | zinc_cqrw1 | zinc_photon2 | not distinguishable | 0.463 | PARTIALLY_COMMENSURABLE | cross-machine; user override 2026-11-09 |
| P2 | zinc_qirw2 | zinc_photon2 | not distinguishable | 0.322 | PARTIALLY_COMMENSURABLE | cross-machine; user override 2026-11-09 |
| C3 | synthetic_quantum | synthetic_degree | not distinguishable | 1 | COMMENSURABLE |  |
| C3 | synthetic_quantum | synthetic_laplacian | synthetic_quantum better | 5.35e-51 | COMMENSURABLE |  |
| C3 | synthetic_quantum | synthetic_rrwp | not distinguishable | 4.55e-13 | COMMENSURABLE |  |
| C3 | synthetic_quantum | synthetic_gcn | synthetic_quantum better | 5.35e-51 | COMMENSURABLE |  |
| C3 | synthetic_laplacian | synthetic_degree | synthetic_degree better | 5.35e-51 | COMMENSURABLE |  |
| C3 | synthetic_gcn | synthetic_degree | synthetic_degree better | 5.35e-51 | COMMENSURABLE |  |
