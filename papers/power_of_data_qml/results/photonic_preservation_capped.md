# Phase 4 preservation criteria [study_photonic_capped.json.gz; coverage COMPLETE]

| Criterion | PA | PB |
|---|---|---|
| P1 | PRESERVED | PRESERVED |
| P2 | PRESERVED | PRESERVED |
| P3 | PRESERVED | PRESERVED |
| P4_ci | PRESERVED | PRESERVED |

- P4 ordering (larger g_PQK has larger advantage): PRESERVED (g order ['PA', 'PB'], advantage order ['PA', 'PB'])
- P5 (bound tracks error): N/A (no bound points at n <= 12) (Spearman nan, bound >= error at nan% of 0 points)

## Details

```json
{
  "PA": {
    "P1": {
      "spearman_n_dFQK": 1.0,
      "d_FQK_max_n": 20,
      "d_FQK_at_max_n_over_N": 0.39152507830394107,
      "max_d_PQK": 1.8583339233108185
    },
    "P2": {
      "17": [
        4.592101862091691,
        1.5530731921881658
      ],
      "18": [
        4.620557704989442,
        1.5441005683622187
      ],
      "19": [
        4.692211401723684,
        1.515472455664867
      ],
      "20": [
        4.671598517732136,
        1.5565049839481544
      ]
    },
    "P3": {
      "n": 20,
      "mean": 0.029056695092626372,
      "ci95": 0.0027251472611792556
    },
    "P4": {
      "20": [
        0.089,
        0.023412081991028653
      ]
    },
    "mean_g_PQK_n_ge_8": 4.644117371634238,
    "mean_advantage_n_ge_8": 0.089
  },
  "PB": {
    "P1": {
      "spearman_n_dFQK": 1.0,
      "d_FQK_max_n": 20,
      "d_FQK_at_max_n_over_N": 0.9907424958203619,
      "max_d_PQK": 2.7568220441673783
    },
    "P2": {
      "17": [
        11.097905067105083,
        1.1425512287227868
      ],
      "18": [
        9.966052614359068,
        1.1431498773327404
      ],
      "19": [
        9.697588479137133,
        1.1357439733075703
      ],
      "20": [
        9.7444545821722,
        1.1191719169851193
      ]
    },
    "P3": {
      "n": 20,
      "mean": 0.002004269289857829,
      "ci95": 0.0010389837526953965
    },
    "P4": {
      "20": [
        0.28200000000000003,
        0.029648741542588567
      ]
    },
    "mean_g_PQK_n_ge_8": 10.12650018569337,
    "mean_advantage_n_ge_8": 0.28200000000000003
  }
}
```

## Photonic vs gate at matched n (mean; [±95% CI] for paired differences; CV protocol)

| n | family | emb | d_FQK | d_PQK | g_FQK | g_PQK | FQK − best MAE | PQK − best acc (eng., N=600) |
|---|---|---|---|---|---|---|---|---|
| 18 | photonic | PA | 290.1 | 1.7 | 1.54 | 4.62 | — | — |
| 18 | photonic | PB | 790.9 | 2.5 | 1.14 | 9.97 | — | — |
| 18 | gate | E2 | 484.4 | 7.2 | 3.77 | 7.01 | — | — |
| 18 | gate | E3 | 782.4 | 18.9 | 1.14 | 10.70 | — | — |
| 20 | photonic | PA | 313.2 | 1.9 | 1.56 | 4.67 | +0.029 [±0.003] | +0.089 [±0.023] |
| 20 | photonic | PB | 792.6 | 2.2 | 1.12 | 9.74 | +0.002 [±0.001] | +0.282 [±0.030] |
| 20 | gate | E2 | 533.1 | 8.0 | 3.77 | 7.08 | +0.058 [±0.033] | +0.064 [±0.014] |
| 20 | gate | E3 | 794.7 | 21.6 | 1.13 | 9.92 | +0.031 [±0.011] | +0.201 [±0.047] |
