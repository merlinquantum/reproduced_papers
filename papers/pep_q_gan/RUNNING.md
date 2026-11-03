# Full Experiment Commands

Run these commands from the repository root after installation. They operate on
one seed/model at a time. In a flat checkout, replace `papers/pep_q_gan/` with
`pep_q_gan/` where needed. The notebook provides the shorter CPU demonstration.

## Sampling and Training

```bash
# Write matched Indist/Dist pools into /absolute/path/to/seed42/.
python implementation.py --paper pep_q_gan --config configs/prior_sampling.json \
  --circuit-seed 42 --pool-root /absolute/path/to/seed42
python implementation.py --paper pep_q_gan --config configs/dist_sampling.json \
  --circuit-seed 42 --pool-root /absolute/path/to/seed42

# Full CUDA training: 100k updates, batch 256, five critic updates per generator update.
python implementation.py --paper pep_q_gan --config configs/training_full.json \
  --dataset-root /absolute/path/to/reference_data --pool /absolute/path/to/seed42/indist

# Dist uses its matching pool and condition.
python implementation.py --paper pep_q_gan --config configs/training_full.json \
  --prior-source dist --dataset-root /absolute/path/to/reference_data \
  --pool /absolute/path/to/seed42/dist

# Gaussian draws fresh latent vectors and needs no photonic pool.
python implementation.py --paper pep_q_gan --config configs/gaussian_baseline.json \
  --dataset-root /absolute/path/to/reference_data
```

The recorded GPU training time was about five hours per model, excluding sampling,
evaluation and queueing. Full CPU training can take substantially longer.
`--dataset-root` specifies the exact directory containing the author files.
When omitted, training resolves `data/pep_q_gan/`; `--data-root` or `DATA_DIR`
overrides the base data directory and appends `pep_q_gan/`.

Full-model generation is available through
`lib.inference.generate_from_checkpoint(checkpoint, data_root, hla_file, output_csv, ...)`.
It checks checkpoint/data/pool identities and preserves all attempts. It requires
a public-format full-model checkpoint; the notebook's small models use their
own visible generation cell.

## MHCflurry Evaluation

MHCflurry 2.3.16 is included in both dependency manifests. Obtain weight release
2.3.0 with the [official downloader](https://openvax.github.io/mhcflurry/model_downloads.html):

```bash
mhcflurry downloads fetch models_class1_presentation --release 2.3.0
mhcflurry downloads path models_class1_presentation --release 2.3.0

# Use the returned bundle path followed by /models.
python implementation.py --paper pep_q_gan --stage evaluate --device cpu \
  --generated-csv /absolute/path/to/generated.csv.gz \
  --hla-file /absolute/path/to/reference_data/unique_hla_allotypes.txt \
  --mhcflurry-models /absolute/path/to/models_class1_presentation/models
```

Input columns are `prior,seed,hla,attempt,peptide`; prior is `gaussian`, `dist`
or `indist`. Supply all 131 HLA and attempts 1–1000 for every included seed/prior.
Valid canonical 8–9-mers are scored without flanks; invalid attempts stay in the
denominator. The screening cutoff is presentation percentile `<0.5`.

Outputs include `predictions.csv.gz`, `per_hla.csv`, `per_seed.csv`, `summary.csv`
and `evaluation_summary.json` under the run directory. Use `--device cuda` only
when an appropriate GPU environment is available. 