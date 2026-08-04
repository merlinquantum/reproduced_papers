# QML Passive Sonar Data

This directory is the shared data root for
`papers/qml_passive_sonar`. Keep only this README and `.gitkeep`
placeholders in Git; audio files are local data artifacts and remain ignored.

## Expected Layout

```text
data/qml_passive_sonar/
|-- shipear/
`-- deepship/
```

Class mapping:

- ShipsEar: `A` fishing/trawlers/tugs, `B` motorboats/sailboats, `C`
  ferries, `D` ocean liners/ro-ro, `E` background.
- DeepShip: `F` cargo, `G` passenger, `H` tanker, `I` tug.

## DeepShip Sample

If a DeepShip run is requested and no local DeepShip `.wav` files are present,
the runner automatically downloads the small public sample from
`github.com/irfankamboh/DeepShip` into `data/qml_passive_sonar/deepship/`.
This sample is enough for CPU smoke and sanity runs, but it is not a
paper-faithful substitute for the full dataset.

To disable that automatic sample download and use synthetic fallback data,
set `"download": false` under the config's `dataset` block.

## Full Datasets

For paper-scale experiments, request or download the original datasets from
their official sources, then create the class folders below and place `.wav`
files into them:

- ShipsEar:
  - Dataset page: <https://underwaternoise.atlanttic.uvigo.es>
  - Access/contact: the dataset page asks users to request full access by
    email at `dsantos(at)gts.uvigo.es`.
  - Paper DOI: <https://doi.org/10.1016/j.apacoust.2016.06.008>
  - After receiving the data, map the released recordings into classes `A`
    through `E`.
- DeepShip:
  - Dataset/sample repository: <https://github.com/irfankamboh/DeepShip>
  - Full-data contact: the repository README says the remaining dataset can
    be requested by email at `mirfan@mail.nwpu.edu.cn`.
  - Paper DOI: <https://doi.org/10.1016/j.eswa.2021.115270>
  - After receiving the data, map cargo/passenger/tanker/tug recordings into
    `F` through `I`.

Once `.wav` files exist locally, the runner uses them directly and does not
download the GitHub sample.
