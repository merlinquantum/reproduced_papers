"""Regenerate the full SEM supervised dataset from ERA5 reanalysis data.

The 3.2 MB full file is **not** stored in git. Smoke tests and the paper
notebook use ``data/bve_qnn/sem_supervised_subset.npz`` (a few KB).

This script documents how to rebuild the full file
``data/bve_qnn/sem_supervised_dataset.npz`` for paper-faithful evaluation
(``configs/example.json``).

The pipeline is documented step-by-step in
``notebooks/neutral_atom/quantum_bve_step_by_step.ipynb``.

Prerequisites
-------------
1. Create a free account at https://cds.climate.copernicus.eu
2. Accept the ERA5 licence terms.
3. Install the CDS API client::

       pip install cdsapi

4. Place your API key in ``~/.cdsapirc``::

       url: https://cds.climate.copernicus.eu/api
       key: <your-uid>:<your-api-key>

Usage
-----
::

    cd papers/bve_qnn
    python utils/generate_dataset.py

Then run the generation notebook. After the full ``.npz`` exists, rebuild
the committed smoke subset with::

    python utils/make_subset.py

The full output file contains:

- ``supervised_features``  — shape (N, 4), columns (t, x, y, z)
- ``supervised_targets``   — shape (N,),   stream-function values
- ``psi_qcl_training``     — SEM reference field for evaluation
- ``lat_downsampled``, ``lon_downsampled`` — grid coordinates

Notes
-----
The original paper does not publish the dataset.  We regenerate it by
downloading ERA5 pressure-level reanalysis (Appendix B) and solving the
barotropic vorticity equation with a spectral-element method at 4°
resolution.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "The full SEM dataset is not committed (too large for this repo).\n"
        "Smoke tests use data/bve_qnn/sem_supervised_subset.npz.\n"
        "\n"
        "To rebuild the full file for paper-faithful evaluation:\n"
        "  1. Set up a Copernicus CDS account (see this file's docstring)\n"
        "  2. Run notebooks/neutral_atom/quantum_bve_step_by_step.ipynb\n"
        "  3. Optionally refresh the smoke subset with:\n"
        "       python utils/make_subset.py\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
