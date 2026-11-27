# lr robustness (seeds 1-3 at every lr; lr 1e-3 from the main sweeps)

| lr | E2 FF | E2 FF+S | E2 TF | TF < FF (seeds) | E5 mean simple / tower / exp / TF | E5 worst simple / tower / exp / TF | TF best mean | TF best worst |
|---|---:|---:|---:|---:|---|---|---|---|
| 1e-3 | 0.0506 | 0.054 | 0.000554 | 3/3 | 0.0242 / 0.0243 / 0.0242 / 0.018 | 0.0642 / 0.0643 / 0.064 / 0.0401 | True | True |
| 3e-3 | 0.0532 | 0.0509 | 0.000643 | 3/3 | 0.0251 / 0.0257 / 0.0259 / 0.00853 | 0.0664 / 0.0672 / 0.0673 / 0.0381 | True | True |
| 1e-2 | 0.0542 | 0.0698 | 0.00198 | 3/3 | 0.0282 / 0.0296 / 0.0374 / 0.0244 | 0.0739 / 0.0749 / 0.0842 / 0.109 | True | False |
