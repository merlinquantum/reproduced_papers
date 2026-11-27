# Data and characterization sources

Generated data belongs under `<shared-data-root>/error-qcbm/`.
See the [README](README.md#how-to-run) for preparation and run commands.

## Synthetic Gaussian target

There is no external Gaussian training dataset. Following the
[author implementation](https://github.com/Quandela/photonic-qcbm), two Gaussian
densities are evaluated at integer coordinates and their sum is normalized.
The 495-bin target uses means 141 and 353 and standard deviations 61. These are
density values at coordinates, not integrals over finite-width bins.

Target bins map to collision-free output states in ascending binary lexicographic
order. The optical simulation includes collisions before detection/conditioning.

## Financial target

The target uses daily European Central Bank USD-per-EUR reference rates,
series `EXR.D.USD.EUR.SP00.A`, requested for 2004-2023.
[Official CSV query](https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2004-01-01&endPeriod=2023-12-31&format=csvdata).

The saved source contains 5,124 valid rates from 2004-01-02 to 2023-12-29.
Successive available observations produce 5,123 natural-log returns, without
interpolation or forward filling. A 495-bin equal-width histogram over the
observed return range defines the target. The provider, dates and bin edges are
reproduction choices because the paper does not specify them fully.

The raw CSV is not bundled. Place the reference CSV at
`<shared-data-root>/error-qcbm/fx/ecb_usd_per_eur_2004_2023.csv` or use the explicit
preparation command in the README. Preparation verifies the reference file;
a changed live API response may be rejected. Verification values remain in
configuration and generated metadata.

Source: European Central Bank. The reproduction authors calculated the log
returns and histogram; these derived results were not produced by the ECB.
See the [ECB reuse conditions](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html).
This target is used for distribution matching, not held-out forecasting.

## Belenos-characteristic simulation

Selected Quandela Hub values retrieved on 2026-10-17 give total transmission 2.8%
and HOM visibility 72.9%. The calibration date was not available; retrieval time
is not a calibration date. The public profile is
[`configs/noise/belenos_20261017.json`](configs/noise/belenos_20261017.json).
The complete Hub response is not distributed; Hub access is not required.

The model uses transmission `T=0.028` and indistinguishability `I=0.729`.
I is passed directly to the backend, whose amplitude-overlap convention uses
`sqrt(I)`. Brightness is fixed at one because T already includes total efficiency.
Other assumptions are `g2=0` and threshold detection. The additional `I=0.84`
control is motivated by the paper's measurement, not a measured Belenos value.

The model excludes dark counts, mode-dependent losses, phase errors, drift and
correlated noise. These are simulations using selected device characteristics,
not QPU measurements or a complete device model.
