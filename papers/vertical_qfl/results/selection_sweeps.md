# Selection and diagnostic sweeps (complete candidate tables)

Metric: final-epoch validation accuracy (20% of the training split), maximised; aggregation: mean (and sample std) over the listed permutation seeds; checkpoint: last epoch; the test split was never used for selection. Generated from each sweep's summary.json.

## T1: `e1_8clients_merlin_tuning_20261019-091645_t1`

Status COMPLETED (COMPLETE SWEEP), 12/12 runs. Selected: [{'client_encoding_scale': 0.7853981633974483, 'lr': 0.01}]; tied within tolerance: [{'client_encoding_scale': 0.7853981633974483, 'lr': 0.01}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"client_encoding_scale": 0.7853981633974483, "lr": 0.01} | 2 / 2 | 0.6000 | 0.0166 |
| {"client_encoding_scale": 0.7853981633974483, "lr": 0.002} | 2 / 2 | 0.4078 | 0.0943 |
| {"client_encoding_scale": 1.5707963267948966, "lr": 0.01} | 2 / 2 | 0.2804 | 0.1026 |
| {"client_encoding_scale": 1.5707963267948966, "lr": 0.002} | 2 / 2 | 0.1529 | 0.0333 |
| {"client_encoding_scale": 3.141592653589793, "lr": 0.01} | 2 / 2 | 0.1078 | 0.0360 |
| {"client_encoding_scale": 3.141592653589793, "lr": 0.002} | 2 / 2 | 0.0941 | 0.0000 |

## T2: `e1_8clients_merlin_tuning_20261019-092947_t2`

Status COMPLETED (COMPLETE SWEEP), 18/18 runs. Selected: [{'client_encoding_scale': 0.39269908169872414, 'lr': 0.01}]; tied within tolerance: [{'client_encoding_scale': 0.39269908169872414, 'lr': 0.01}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"client_encoding_scale": 0.39269908169872414, "lr": 0.01} | 2 / 2 | 0.6706 | 0.0499 |
| {"client_encoding_scale": 0.19634954084936207, "lr": 0.01} | 2 / 2 | 0.6529 | 0.0471 |
| {"client_encoding_scale": 0.7853981633974483, "lr": 0.01} | 2 / 2 | 0.6000 | 0.0166 |
| {"client_encoding_scale": 0.39269908169872414, "lr": 0.03} | 2 / 2 | 0.1235 | 0.0250 |
| {"client_encoding_scale": 0.7853981633974483, "lr": 0.03} | 2 / 2 | 0.1157 | 0.0083 |
| {"client_encoding_scale": 0.19634954084936207, "lr": 0.03} | 2 / 2 | 0.1039 | 0.0305 |
| {"client_encoding_scale": 0.19634954084936207, "lr": 0.1} | 2 / 2 | 0.1039 | 0.0028 |
| {"client_encoding_scale": 0.39269908169872414, "lr": 0.1} | 2 / 2 | 0.0980 | 0.0111 |
| {"client_encoding_scale": 0.7853981633974483, "lr": 0.1} | 2 / 2 | 0.0882 | 0.0139 |

## D1: `e1_8clients_merlin_diagnostic_20261020-065450_d1`

Status COMPLETED (COMPLETE SWEEP), 16/16 runs. Selected: [{'epochs': 100, 'server_head': 1, 'server_modes': 8}]; tied within tolerance: [{'epochs': 100, 'server_head': 1, 'server_modes': 8}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"epochs": 100, "server_head": 1, "server_modes": 8} | 2 / 2 | 0.8216 | 0.0305 |
| {"epochs": 30, "server_head": 1, "server_modes": 12} | 2 / 2 | 0.8000 | 0.0166 |
| {"epochs": 100, "server_head": 1, "server_modes": 12} | 2 / 2 | 0.7882 | 0.0222 |
| {"epochs": 100, "server_head": 0, "server_modes": 12} | 2 / 2 | 0.7608 | 0.0055 |
| {"epochs": 100, "server_head": 0, "server_modes": 8} | 2 / 2 | 0.7529 | 0.0610 |
| {"epochs": 30, "server_head": 0, "server_modes": 12} | 2 / 2 | 0.7373 | 0.0222 |
| {"epochs": 30, "server_head": 1, "server_modes": 8} | 2 / 2 | 0.7235 | 0.0749 |
| {"epochs": 30, "server_head": 0, "server_modes": 8} | 2 / 2 | 0.6706 | 0.0499 |

## D2: `e1_8clients_merlin_diagnostic_d2_20261020-074747_d2`

Status COMPLETED (COMPLETE SWEEP), 16/16 runs. Selected: [{'n_photons': 4, 'reuploads': 1, 'server_head': 2}]; tied within tolerance: [{'n_photons': 4, 'reuploads': 1, 'server_head': 2}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"n_photons": 4, "reuploads": 1, "server_head": 2} | 2 / 2 | 0.8196 | 0.0111 |
| {"n_photons": 4, "reuploads": 2, "server_head": 2} | 2 / 2 | 0.7255 | 0.1054 |
| {"n_photons": 4, "reuploads": 1, "server_head": 0} | 2 / 2 | 0.7137 | 0.0388 |
| {"n_photons": 4, "reuploads": 2, "server_head": 0} | 2 / 2 | 0.1333 | 0.0388 |
| {"n_photons": 8, "reuploads": 1, "server_head": 0} | 2 / 2 | 0.1020 | 0.0000 |
| {"n_photons": 8, "reuploads": 1, "server_head": 2} | 2 / 2 | 0.1020 | 0.0000 |
| {"n_photons": 8, "reuploads": 2, "server_head": 2} | 2 / 2 | 0.1020 | 0.0000 |
| {"n_photons": 8, "reuploads": 2, "server_head": 0} | 2 / 2 | 0.1000 | 0.0028 |

## D3: `e1_8clients_merlin_diagnostic_d3_20261020-100305_d3`

Status COMPLETED (COMPLETE SWEEP), 8/8 runs. Selected: [{'n_photons': 8, 'server_head': 2}]; tied within tolerance: [{'n_photons': 8, 'server_head': 2}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"n_photons": 8, "server_head": 2} | 2 / 2 | 0.8216 | 0.0638 |
| {"n_photons": 4, "server_head": 2} | 2 / 2 | 0.7824 | 0.0749 |
| {"n_photons": 4, "server_head": 0} | 2 / 2 | 0.7686 | 0.0388 |
| {"n_photons": 8, "server_head": 0} | 2 / 2 | 0.5373 | 0.1886 |

## D4: `e1_8clients_merlin_diagnostic_d4_20261030-152709_d4`

Status COMPLETED (COMPLETE SWEEP), 8/8 runs. Selected: [{'circuit': {'client_feature_maps': ['identity', 'square', 'cos'], 'message_width': 8, 'server_photons': 4}, 'n_client_stages': 8, 'n_modes': 12}]; tied within tolerance: [{'circuit': {'client_feature_maps': ['identity', 'square', 'cos'], 'message_width': 8, 'server_photons': 4}, 'n_client_stages': 8, 'n_modes': 12}, {'circuit': {'client_feature_maps': ['identity', 'square', 'cos'], 'first_layer': 'superposition', 'message_width': 8, 'server_photons': 4}, 'n_client_stages': 8, 'n_modes': 12}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"circuit": {"client_feature_maps": ["identity", "square", "cos"], "first_layer": "superposition", "message_width": 8, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.8353 | 0.0111 |
| {"circuit": {"client_feature_maps": ["identity", "square", "cos"], "message_width": 8, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.8333 | 0.0083 |
| {"circuit": {"client_feature_maps": ["identity", "square", "cos"], "first_layer": "superposition", "message_width": 8, "server_feature_maps": ["identity", "square", "cos"], "server_photons": 4}, "n_client_stages": 8, "n_modes": 12, "server_modes": 12} | 2 / 2 | 0.7765 | 0.0555 |
| {"circuit": {"first_layer": "superposition"}} | 2 / 2 | 0.7275 | 0.0360 |

## D5: `e1_8clients_merlin_diagnostic_d5_20261030-172715_d5`

Status COMPLETED (COMPLETE SWEEP), 8/8 runs. Selected: [{'circuit': {'client_feature_maps': ['identity', 'square', 'cos'], 'client_mixer': 'superposition', 'message_width': 8, 'mixer_depth': 2, 'server_photons': 4}, 'n_client_stages': 8, 'n_modes': 12}]; tied within tolerance: [{'circuit': {'client_feature_maps': ['identity', 'square', 'cos'], 'client_mixer': 'superposition', 'message_width': 8, 'mixer_depth': 2, 'server_photons': 4}, 'n_client_stages': 8, 'n_modes': 12}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"circuit": {"client_feature_maps": ["identity", "square", "cos"], "client_mixer": "superposition", "message_width": 8, "mixer_depth": 2, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.8039 | 0.0333 |
| {"circuit": {"client_feature_maps": ["identity", "identity", "identity"], "client_mixer": "superposition", "message_width": 8, "mixer_depth": 1, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.7882 | 0.0111 |
| {"circuit": {"client_feature_maps": ["identity", "identity", "identity"], "client_mixer": "superposition", "message_width": 8, "mixer_depth": 2, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.7745 | 0.0083 |
| {"circuit": {"client_feature_maps": ["identity", "square", "cos"], "client_mixer": "superposition", "message_width": 8, "mixer_depth": 1, "server_photons": 4}, "n_client_stages": 8, "n_modes": 12} | 2 / 2 | 0.7647 | 0.0388 |

## C-T: `classical_tuning_20261020-144439_ct`

Status COMPLETED (COMPLETE SWEEP), 48/48 runs. The rule was applied separately per group (architecture, size) by `utils/select_hyperparameters.py --group-by architecture,size` (tie tolerance 0.005, ties to the smallest lr); source `results/classical_selection.json`.

### C-T group {"architecture": "central", "size": "gate"}

Selected: {'architecture': 'central', 'central_hidden': 270, 'client_hidden': 195, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 30, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 107, 'size': 'gate'}; tied within tolerance: [{'architecture': 'central', 'central_hidden': 270, 'client_hidden': 195, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 30, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 107, 'size': 'gate'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "central", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9314 | 0.0028 |
| {"architecture": "central", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9216 | 0.0222 |
| {"architecture": "central", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9059 | 0.0000 |
| {"architecture": "central", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9039 | 0.0028 |

### C-T group {"architecture": "vertical", "size": "gate"}

Selected: {'architecture': 'vertical', 'central_hidden': 270, 'client_hidden': 195, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 30, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 107, 'size': 'gate'}; tied within tolerance: [{'architecture': 'vertical', 'central_hidden': 270, 'client_hidden': 195, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 30, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 107, 'size': 'gate'}, {'architecture': 'vertical', 'central_hidden': 270, 'client_hidden': 195, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 30, 'lr': 0.005, 'server_encoding': 'server_wide', 'server_hidden': 107, 'size': 'gate'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "vertical", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9098 | 0.0222 |
| {"architecture": "vertical", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.9078 | 0.0139 |
| {"architecture": "vertical", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.8941 | 0.0166 |
| {"architecture": "vertical", "central_hidden": 270, "client_hidden": 195, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 30, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 107, "size": "gate"} | 2 / 2 | 0.8745 | 0.0166 |

### C-T group {"architecture": "central", "size": "photonic_f"}

Selected: {'architecture': 'central', 'central_hidden': 10, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 7, 'size': 'photonic_f'}; tied within tolerance: [{'architecture': 'central', 'central_hidden': 10, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 7, 'size': 'photonic_f'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "central", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.9059 | 0.0000 |
| {"architecture": "central", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.9000 | 0.0028 |
| {"architecture": "central", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.9000 | 0.0028 |
| {"architecture": "central", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.8980 | 0.0055 |

### C-T group {"architecture": "central", "size": "photonic_h"}

Selected: {'architecture': 'central', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.001, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}; tied within tolerance: [{'architecture': 'central', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.001, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}, {'architecture': 'central', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "central", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.9020 | 0.0000 |
| {"architecture": "central", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.9000 | 0.0139 |
| {"architecture": "central", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8902 | 0.0055 |
| {"architecture": "central", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8902 | 0.0111 |

### C-T group {"architecture": "vertical", "size": "photonic_h"}

Selected: {'architecture': 'vertical', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}; tied within tolerance: [{'architecture': 'vertical', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}, {'architecture': 'vertical', 'central_hidden': 12, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.005, 'server_encoding': 'server_wide', 'server_hidden': 12, 'size': 'photonic_h'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "vertical", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8667 | 0.0166 |
| {"architecture": "vertical", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8627 | 0.0166 |
| {"architecture": "vertical", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8490 | 0.0250 |
| {"architecture": "vertical", "central_hidden": 12, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 12, "size": "photonic_h"} | 2 / 2 | 0.8451 | 0.0083 |

### C-T group {"architecture": "vertical", "size": "photonic_f"}

Selected: {'architecture': 'vertical', 'central_hidden': 10, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 7, 'size': 'photonic_f'}; tied within tolerance: [{'architecture': 'vertical', 'central_hidden': 10, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.002, 'server_encoding': 'server_wide', 'server_hidden': 7, 'size': 'photonic_f'}, {'architecture': 'vertical', 'central_hidden': 10, 'client_hidden': 7, 'client_slices': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 100, 'lr': 0.01, 'server_encoding': 'server_wide', 'server_hidden': 7, 'size': 'photonic_f'}].

| Candidate | Successful / expected | Mean | Std |
|---|---:|---:|---:|
| {"architecture": "vertical", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.01, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.8471 | 0.0222 |
| {"architecture": "vertical", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.002, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.8431 | 0.0499 |
| {"architecture": "vertical", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.001, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.8392 | 0.0055 |
| {"architecture": "vertical", "central_hidden": 10, "client_hidden": 7, "client_slices": [0, 1, 2, 3, 4, 5, 6, 7], "epochs": 100, "lr": 0.005, "server_encoding": "server_wide", "server_hidden": 7, "size": "photonic_f"} | 2 / 2 | 0.8157 | 0.0333 |
