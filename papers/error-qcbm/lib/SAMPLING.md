# Optical backend and sampling

## Probability flow

The 12-mode/four-photon input is `[0,0,1,0,1,0,1,0,1,0,0,0]`.
The Clements mesh has 66 MZIs and 132 trainable phases. Perceval describes the
circuit and detectors; MerLin computes the optical evolution and native shots.

```text
Input and phases -> optical evolution and noise -> threshold detection
                 -> N native shots -> integer counts
                 -> post-selection or recycling -> finite-shot loss
```

Physical evolution includes collisions. The fixed four-photon Fock space has
1,365 outcomes, while the target uses 495 collision-free patterns. Threshold
detectors register one click for any positive occupation, so fewer than four
clicks do not necessarily imply photon loss. Vacuum and discarded events count
toward N. Empty required sectors fail without extra draws or fallback estimates.

## Sampling modes

Ordinary configs use `training.native_count_accumulation = "python"`.
The optional `"vectorized"` adapter preserves native multinomial draws and
replaces count accumulation with `torch.bincount` via a temporary sampler hook.
It is validated for CPU float64, one Torch thread and g2=0. Use separate
processes for independent seeds; changing chunk sizes can change random streams.

Optimized resumable runs require matching validation evidence. Generate it with:

```bash
python implementation.py --paper error-qcbm --config configs/sampler_validation_full.json
```

The full check uses 50M shots per sampling call and writes
`outdir/sampler_validation/evidence.json`. In the experiment config, set
`native_count_accumulation` to `"vectorized"`, `native_chunk_shots` to `1000000`,
`torch_threads` to `1`, and `vectorized_validation_file` to that evidence path,
all under `training`. Regenerate evidence after changing the environment or
sampler code. The smaller `sampler_validation_smoke.json` is a quick check only.

See [TRAINING.md](TRAINING.md) for finite-shot optimization and shot budgets.

## Dependency notices

<details>
<summary>MerLin and Perceval license notices</summary>

### MerLin

```text
MIT License

Copyright (c) 2025 Quandela

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Perceval

```text
MIT License

Copyright (c) 2022 Quandela

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

As a special exception, the copyright holders of exqalibur library give you
permission to combine exqalibur with code included in the standard release of
Perceval under the MIT license (or modified versions of such code). You may
copy and distribute such a combined system following the terms of the MIT
license for both exqalibur and Perceval. This exception for the usage of
exqalibur is limited to the python bindings used by Perceval.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
