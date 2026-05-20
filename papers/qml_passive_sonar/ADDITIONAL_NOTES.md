# Additional Notes

This file collects implementation and reproduction notes that are useful for
extending the QML passive-sonar scaffold beyond the smoke configuration.

## Implementation Choices

- The qubit HQ-CNN uses a pure-PyTorch statevector simulator in
  `lib/quantum.py`. Gradients flow through torch autograd; the forward map is
  the same as the paper's RY/RZ/ring-CNOT circuit, while training does not use
  parameter-shift gradients.
- The MerLin photonic variant in `lib/models_merlin.py` uses a generic
  rectangular interferometer with trainable beam-splitter and phase-shifter
  parameters, per-mode input phase encoding, and a readout interferometer.
  The CPU-feasible default is 6 modes / 2 photons in the unbunched
  computation space.
- Synthetic fallback data is intentionally class-structured: non-background
  classes use class-specific tonal sets with harmonics and coloured noise.
  That makes SAV/DEMON detection and classifier smoke tests meaningful even
  without ShipsEar access.
- `model.fc_dim` is configurable. Paper-aligned configs use the 4096-dim CNN
  bottleneck; CPU sanity configs reduce this to 64 or 128 to keep memory and
  runtime manageable.

## Dataset Notes

- ShipsEar requires registration/contact with the dataset maintainers. The
  default smoke path therefore uses synthetic ShipsEar-like data.
- DeepShip CPU sanity runs can use the small public GitHub sample. The runner
  downloads it automatically only when a DeepShip run is requested and no
  local DeepShip `.wav` files are present.
- The small DeepShip sample is useful for a quick real-audio sanity check, but
  it is not statistically meaningful for paper-level accuracy claims. Use the
  full dataset for serious evaluation.

## Measured CPU Sanity Results

The following numbers were obtained with seed 42, 20 epochs, `lr=5e-4`,
batch size 8, 8-second frames, 4 kHz target sampling rate, 64x64 images, and
`fc_dim=128`.

| Run | Final test accuracy | Train-test gap |
|-----|--------------------:|---------------:|
| SAV vs DEMON detection, synthetic ShipsEar | 100.0% detection; 4 vs 29 false peaks | n/a |
| CNN, synthetic ShipsEar | 100.0% | 0.000 |
| HQ-CNN, synthetic ShipsEar | 100.0% | 0.004 |
| MerLin, synthetic ShipsEar | 100.0% | 0.028 |
| CNN, DeepShip sample | 92.98% | 0.062 |
| HQ-CNN, DeepShip sample | 89.47% | 0.074 |
| MerLin, DeepShip sample | 92.98% | 0.031 |

The CNN baseline on the small DeepShip sample is close to the paper's DeepShip
CNN value, but this should be read only as a sanity check: the sample has 8
audio files, reduced image/backbone sizes, 20 epochs instead of 100, and a
frame-level random split rather than a source-stratified split.

The MerLin variant matched the CNN baseline on the small DeepShip sample while
showing a smaller train-test gap. This is qualitatively consistent with the
paper's generalization-gap claim, but the result needs full-data,
source-stratified evaluation before drawing conclusions.

## Scaling Toward Paper-Level Runs

1. Place full ShipsEar and/or DeepShip WAV files under
   `data/qml_passive_sonar/<dataset>/<class>/`; see
   `data/qml_passive_sonar/README.md` for access links and class mapping.
2. Use `configs/classification_shipear.json` or
   `configs/classification_deepship.json` for paper-aligned qubit HQ-CNN runs
   with `fc_dim=4096`, `image_size=224`, and 100 epochs.
3. Use a GPU for paper-aligned configs. The CPU configs are intentionally
   reduced.
4. Split real recordings at the source/vessel level before training if you
   need a paper-faithful source-stratified evaluation.
5. For photonic experiments, increase `n_modes` beyond the CPU-friendly
   6-mode setting only after checking memory and runtime.
