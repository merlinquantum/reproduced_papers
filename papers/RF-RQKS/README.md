# RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks - Reproduction

## Dataset construction

The first reproduction stage builds the paper's binary anomaly-classification dataset from measured LTE IQ captures. The measured LTE waveform is always the normal signal; the builder does not synthesize a replacement LTE background.

Each unique LTE segment contains 1,300,000 complex IQ points sampled at 61.44 MHz (21.15 ms). The segment is emitted twice:

- label `0`: unchanged LTE signal;
- label `1`: the same LTE signal with one chirp, barrage-jamming, or frequency-hopping-noise anomaly.

The anomaly is power-scaled to one JSR from `[-10, -8, -6, -4, -2, 0, 2, 5]` dB. Train and test partitions contain disjoint LTE source segments. The production configuration produces 21,600 training samples and 8,124 test samples.

The paper specifies a Hann-window STFT with a 3,250-point window, an 8,192-point FFT, and 25% overlap. The central 48 MHz is retained and resized to a 400-by-400 frequency-by-time array.

### Required source data

Download and extract the measured IQ portion of the [WASD dataset](https://ieee-dataport.org/open-access/wasd-wireless-anomaly-signal-dataset) under the repository data root, for example:

```text
data/RF-RQKS/raw/
`-- <band directories>/
    `-- *.bin
```

Binary files must contain interleaved `int32` values in `[I0, Q0, I1, Q1, ...]` order. Files may contain one or several non-overlapping 1,300,000-point segments. Trailing values shorter than a complete segment are ignored. The builder fails if fewer than 14,862 complete segments are available.

### Generate the full dataset

From `papers/RF-RQKS`:

```bash
python utils/generate_dataset.py \
  --input-root ../../data/RF-RQKS/raw \
  --output-root ../../data/RF-RQKS/processed
```

The authoritative parameters are in `configs/dataset.json`. Generation is deterministic for a fixed input tree and seed. The output directory must not already exist.

```text
data/RF-RQKS/processed/
|-- manifest.json
|-- plot_data_examples.py
|-- train/
|   |-- spectrograms.npy  # (21600, 400, 400), float32
|   |-- labels.npy        # (21600,), uint8
|   `-- metadata.csv
`-- test/
    |-- spectrograms.npy  # (8124, 400, 400), float32
    |-- labels.npy        # (8124,), uint8
    `-- metadata.csv
```

The arrays are written incrementally as NumPy memory-mapped files. At `float32`, the two spectrogram arrays require about 17.7 GiB in total, excluding temporary filesystem overhead.

For the independently downloaded `36_LTE_1` band, generate a complete single-band subset from all 505 captures:

```bash
python utils/generate_dataset.py \
  --input-root ../../data/RF-RQKS/raw/36_LTE_1 \
  --output-root ../../data/RF-RQKS/processed_36_lte_1 \
  --config configs/dataset_36_lte_1.json
```

This produces 800 training samples and 210 test samples after pairing every LTE capture with its anomaly-injected version. The spectrogram arrays require approximately 617 MiB. This subset is suitable for pipeline development and single-band experiments, but it does not reproduce the paper's full multi-band sample count.

### Match the paper's ablation dataset size

When only the 36_LTE_1 processed subset is available, expand it to the paper's
ablation row counts with paired spectrogram translations:

```bash
python utils/augment_dataset.py \
  --input-root ../../data/RF-RQKS/processed_36_lte_1 \
  --output-root ../../data/RF-RQKS/processed_ablation_paper_size
```

The default output contains 10,800 training pairs (21,600 rows) and 4,062 test
pairs (8,124 rows). Each normal/anomaly pair is sampled with replacement and
receives the same bounded frequency/time translation, so the pair remains
intact for the grouped ablation split. `metadata.csv` records the original
`source_pair_index` and translations. Use the resulting directory as the
input to `utils/generate_dct_representation.py`.

This is a row-count matching procedure for the ablation study, not a substitute
for collecting the paper's independent multi-band LTE captures. The manifest
records this limitation; use the full dataset-generation command above when
the complete WASD source data is available.

#### Why augmentation is used

The original WASD IQ collection is larger than 200 GB. Rebuilding the complete
paper dataset from raw IQ therefore requires substantial storage and repeated
STFT computation. The augmentation path is a practical ablation-study path
when only the 36_LTE_1 subset is available:

1. Generate the 400 training and 105 test source pairs from the available IQ
   captures.
2. Sample those complete pairs with replacement until the paper's 10,800
   training and 4,062 test source-pair counts are reached.
3. Apply one random integer frequency translation and one random integer time
   translation to each pair. The default maximum translation is 10% of the
   corresponding spectrogram dimension.
4. Apply exactly the same translation to the normal and anomalous member of a
   pair, preserving the binary label relationship.
5. Write new spectrogram arrays and provenance metadata. The metadata records
   both the original `source_pair_index` and the applied translations.

Augmentation is performed on the 400-by-400 log-magnitude spectrograms rather
than on the raw IQ files. This avoids duplicating the more than 200 GB raw
collection and avoids rerunning the expensive anomaly synthesis and STFT for
every duplicate. The resulting paper-sized spectrogram dataset still requires
approximately 17.7 GiB at `float32`, so it should be generated once and then
reused. It contains repeated views of the available source captures, not new
independent LTE measurements; this distinction must be considered when
interpreting ablation results.

After generation, plot representative processed samples directly from the dataset directory:

```bash
cd data/RF-RQKS/processed
python plot_data_examples.py
python plot_data_examples.py --split test --output test_examples.png
```

The script memory-maps `spectrograms.npy`, selects examples from `metadata.csv`, and writes `train_examples.png` or `test_examples.png` without loading the full split into memory. The normal panel and chirp panel use the same measured LTE source pair.

### Reproduction boundary

The RF-RQKS paper specifies the dataset dimensions, STFT settings, anomaly families, random placement, and JSR values. It does not specify the exact probability distributions for anomaly duration, chirp endpoints, hopping bandwidth, or dwell time, nor the image-resize interpolation. This implementation exposes those choices in `configs/dataset.json` and records realized parameters per sample in `metadata.csv`. The defaults are a documented reproduction choice, not a claim about unpublished author code.

### Verification figure

Generate the four-panel anomaly comparison with a measured IQ segment:

```bash
python utils/plot_dataset_examples.py \
  --iq-path ../../data/RF-RQKS/raw/<band>/<capture>.bin
```

When the WASD captures are not yet available, exercise the same anomaly and STFT pipeline with an explicitly synthetic LTE-like background:

```bash
python utils/plot_dataset_examples.py --synthetic-demo
```

The synthetic option is for implementation verification only and is not used to build the reproduced dataset.

![RF-RQKS dataset pipeline verification](results/rf_rqks_dataset_examples.png)

## Representation generation

The ablation does not load the 400-by-400 spectrograms directly. The converter
reads each processed split in batches, computes a two-dimensional DCT-II, keeps
the low-frequency coefficient block, and writes a compact feature cache. Run
the converter after either the full dataset or the augmented dataset has been
created. For the augmented ablation dataset:

```bash
python utils/generate_dct_representation.py \
  --input-root ../../data/RF-RQKS/processed_ablation_paper_size \
  --output-root ../../data/RF-RQKS/representations/dct64x64_ablation_paper_size
```

For the full dataset, use:

```bash
python utils/generate_dct_representation.py \
  --input-root ../../data/RF-RQKS/processed \
  --output-root ../../data/RF-RQKS/representations/dct64x64
```

For each 400-by-400 spectrogram, this applies a separable two-dimensional DCT-II
with `norm="ortho"`, retains the upper-left 64-by-64 low-index coefficient
block, and flattens it to 4,096 features. The cache for the paper-sized dataset
occupies approximately 465 MiB at `float32` and has this layout:

```text
data/RF-RQKS/representations/dct64x64/
|-- manifest.json
|-- train/
|   |-- features.npy  # (21600, 4096), float32
|   |-- labels.npy
|   `-- metadata.csv
`-- test/
    |-- features.npy  # (8124, 4096), float32
    |-- labels.npy
    `-- metadata.csv
```

The cached coefficients are intentionally unnormalized. For Stages 1-4, first create the locked ablation training and validation indices, call `fit_feature_standardization` using only the ablation training indices, and apply the resulting statistics to both partitions with `standardize_features`. For Stage 5, refit the statistics using every row of the raw training split before transforming the train and test features. Fitting normalization on all 21,600 training rows before the Stage 1-4 split would leak validation information.

The converter copies the original metadata into the representation cache. This
keeps the augmentation provenance available while the ablation operates only
on `features.npy`. The 128-by-128 variant can be generated similarly, but it
requires substantially more feature storage and model computation.

Generate the 128-by-128 DCT variant with the same converter and its dedicated configuration:

```bash
python utils/generate_dct_representation.py \
  --input-root ../../data/RF-RQKS/processed \
  --output-root ../../data/RF-RQKS/representations/dct128x128 \
  --config configs/dct128x128.json
```

This representation contains 16,384 flattened coefficients per sample and requires approximately 1.81 GiB for both `float32` feature arrays.

## Runnable ablation study

The five-stage experiment protocol from the local `thales-qks` project is
integrated with the repository runtime. It performs:

1. a mode-count, episode-count, and entanglement sweep;
2. a depth sweep over the Stage 1 shortlist;
3. a matched depth/episode sweep;
4. a photon-count sweep; and
5. held-out comparison of four direct and QKS readouts.

The loader splits by the generated `pair_index`, ensuring that the normal and
anomaly-injected versions of one source pair cannot occur in different
partitions. The reduced `ablation_36_lte_1.json` configuration reserves 15% of
the 10,800 augmented training pairs for validation: 18,360 training rows and
3,240 validation rows. Stages 1-4 use DCT standardization fitted only on the
resulting training partition. Stage 5 fits fresh statistics on all 21,600
development rows before evaluating the separate 8,124-row test set.

Install the exact MerLin version used by the photonic port:

```bash
cd papers/RF-RQKS
python -m pip install -r requirements.txt
cd ../..
```

Run the minimal photonic integration check on the generated LTE dataset:

```bash
papers/RF-RQKS/venv/bin/python implementation.py \
  --paper RF-RQKS \
  --config papers/RF-RQKS/configs/photonic_smoke_36_lte_1.json
```

For the paper-sized augmented representation, override the representation path
when running the reduced ablation:

```bash
python implementation.py \
  --paper RF-RQKS \
  --config papers/RF-RQKS/configs/ablation_36_lte_1.json \
  --representation-path representations/dct64x64_ablation_paper_size
```

The representation path is relative to `data/RF-RQKS/`. Passing the processed
spectrogram directory here is incorrect; the ablation requires the generated
`features.npy`, `labels.npy`, and `metadata.csv` files.

The original `configs/ablation_36_lte_1.json` configuration remains useful for
the unaugmented 36_LTE_1 experiment. Its default representation path points
to `representations/dct64x64_36_lte_1` and therefore uses 800 development rows
and 210 test rows.

The second command uses the photonic sampler and can be computationally
expensive. First estimate runtime with `photonic_smoke_36_lte_1.json`. The
unmodified Thales-QKS sweep grid is available as
`configs/full_ablation.json`; it is substantially more expensive and
does not turn this single LTE band into the paper's full multi-band dataset.

Running without a named config executes a fast classical random-feature smoke
test of all five stages:

```bash
python implementation.py --paper RF-RQKS
```

Every invocation creates `outdir/run_YYYYMMDD-HHMMSS/` containing
`results.json`, `summary.json`, the resolved `config_snapshot.json`, and the
ablation figures under `figures/`.

The ablation figures use the same layouts as the original `thales-qks`
protocol: Stage 1 entangling-on/off AUROC heatmaps, Stage 2 depth curves,
Stage 3 matched-depth/episode grouped bars, Stage 4 photon-count AUROC/F1
curves, and the Stage 5 direct-versus-photonic readout comparison. They are
written as `stage_1_validation_auroc_*.png`,
`stage_2_validation_auroc_depth.png`, `stage_3_validation_scores.png`,
`stage_4_validation_scores_photons.png`, and
`stage_5_regression_functions.png`.

## Tests

From `papers/RF-RQKS`:

```bash
./venv/bin/pytest -q
```
