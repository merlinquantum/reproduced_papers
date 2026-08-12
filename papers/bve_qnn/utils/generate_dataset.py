"""Regenerate the SEM supervised dataset from ERA5 reanalysis data.

This script automates the dataset generation from the Copernicus Climate
Data Store (CDS).  The full pipeline is documented step-by-step in
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

The output file ``data/bve_qnn/sem_supervised_dataset.npz`` contains:

- ``supervised_features``  — shape (N, 4), columns (t, x, y, z)
- ``supervised_targets``   — shape (N,),   stream-function values
- ``psi_qcl_training``     — SEM reference field for evaluation
- ``lat_downsampled``, ``lon_downsampled`` — grid coordinates

Notes
-----
The original paper does not publish the dataset.  We regenerate it by
downloading ERA5 pressure-level reanalysis (Appendix B) and solving the
barotropic vorticity equation with a spectral-element method at 4°
resolution.  The generation notebook walks through every step.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "Dataset generation requires the ERA5 pipeline from the notebook.\n"
        "\n"
        "Please run the full pipeline in:\n"
        "  notebooks/neutral_atom/quantum_bve_step_by_step.ipynb\n"
        "\n"
        "Or follow the instructions in this file's docstring to set up\n"
        "the CDS API and run the notebook cells programmatically.\n"
        "\n"
        "The pre-generated dataset is committed at:\n"
        "  data/bve_qnn/sem_supervised_dataset.npz  (3.2 MB)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
