"""Data construction for the VQNN experiments.

No experiment uses downloaded data. Two assets are generated procedurally,
following the original authors' procedures (LOG.md, "Data Acquisition Log"):

- E4 classification: the "circle" data set of the virtual quantum classifier
  (uniform points in ``[-1, 1]^2``; label 0 inside radius 0.7, label 1
  outside; 200 training and 100 test points).
- E5 ground state: the qubit-tapered HeH+ Hamiltonian built with
  PennyLane ``qml.qchem`` (STO-3G, Jordan-Wigner, 4 -> 2 qubits).

A pre-generated classification data set can instead be loaded from the shared
data root (``DATA_DIR`` / ``--data-root``) with ``data.source = "file"``.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .settings import real_dtype

logger = logging.getLogger(__name__)

PAPER_NAME = "virtual_qnn"
CIRCLE_RADIUS = 0.7
BOHR_TO_ANGSTROM = 0.529177210903


def generate_circle_dataset(
    n_train: int, n_test: int, seed_data: int, boundary_gap: float = 0.0
) -> dict[str, np.ndarray]:
    """Generate the circle classification data set of paper Sec. III B.

    The rejection-sampling and reseeding scheme reproduces the authors' rule:
    before each candidate draw NumPy is reseeded with
    ``(seed_data + 10) * 1000 + n_rejected + n_accepted``, a point uniform in
    ``[-1, 1]^2`` is drawn, and it is labelled 0 if its norm is below
    ``0.7 - gap/2``, 1 if above ``0.7 + gap/2``, and rejected otherwise.

    Parameters
    ----------
    n_train, n_test : int
        Number of training and test points (paper: 200 and 100).
    seed_data : int
        Data-set seed. The paper's Fig. 6d seeds were not recorded.
    boundary_gap : float
        Width of the excluded annulus around the decision circle (paper: 0).
        Default value is 0.0.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``train_x`` ``(n_train, 2)``, ``train_y`` ``(n_train,)``, ``test_x``,
        ``test_y``; labels are 0.0 or 1.0.
    """
    points, labels = [], []
    n_rejected = 0
    while len(points) < n_train + n_test:
        np.random.seed((seed_data + 10) * 1000 + n_rejected + len(points))
        candidate = np.random.rand(2) * 2 - 1
        norm = np.linalg.norm(candidate)
        if norm < CIRCLE_RADIUS - boundary_gap / 2:
            points.append(candidate)
            labels.append(0.0)
        elif norm > CIRCLE_RADIUS + boundary_gap / 2:
            points.append(candidate)
            labels.append(1.0)
        else:
            n_rejected += 1
    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return {
        "train_x": x[:n_train],
        "train_y": y[:n_train],
        "test_x": x[n_train:],
        "test_y": y[n_train:],
    }


def paper_data_dir(cfg: dict) -> Path:
    """Return ``<data_root>/virtual_qnn`` from the resolved config.

    Raises
    ------
    KeyError
        If ``cfg["data_root"]`` is missing (the shared runtime always sets it).
    """
    return Path(cfg["data_root"]) / PAPER_NAME


def load_classification_arrays(cfg: dict) -> dict[str, np.ndarray]:
    """Build or load the classification arrays described by ``cfg["data"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with a ``data`` section containing ``source``
        (``"generated"`` or ``"file"``) and the generator or file settings.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays as returned by :func:`generate_circle_dataset`.

    Raises
    ------
    ValueError
        If ``data.source`` is unknown.
    FileNotFoundError
        If ``data.source == "file"`` and the file does not exist.
    """
    data_cfg = cfg["data"]
    source = data_cfg["source"]
    if source == "generated":
        return generate_circle_dataset(
            n_train=int(data_cfg["n_train"]),
            n_test=int(data_cfg["n_test"]),
            seed_data=int(data_cfg["seed_data"]),
            boundary_gap=float(data_cfg["boundary_gap"]),
        )
    if source == "file":
        path = paper_data_dir(cfg) / data_cfg["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Classification data file not found: {path}")
        with np.load(path) as archive:
            return {
                key: archive[key] for key in ("train_x", "train_y", "test_x", "test_y")
            }
    raise ValueError(f"Unknown data.source {source!r}; expected 'generated' or 'file'")


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Return a full-batch loader over one split of the classification data.

    The reference training loop is full-batch gradient descent, so the default
    batch size is the split size.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``data`` section, top-level ``dtype``).
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(x, y)`` with ``x`` of shape ``(batch, 2)`` and ``y`` of shape
        ``(batch,)``.

    Raises
    ------
    ValueError
        If ``split`` is not ``"train"`` or ``"test"``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'")
    arrays = load_classification_arrays(cfg)
    dtype = real_dtype(cfg)
    x = torch.as_tensor(arrays[f"{split}_x"], dtype=dtype)
    y = torch.as_tensor(arrays[f"{split}_y"], dtype=dtype)
    batch_size = cfg["data"].get("batch_size") or len(x)
    return DataLoader(TensorDataset(x, y), batch_size=int(batch_size), shuffle=False)


@cache
def heh_tapered_hamiltonian(half_distance_bohr: float) -> dict:
    """Build the 2-qubit tapered HeH+ Hamiltonian used by the VVQE experiment.

    The construction mirrors the authors' reference notebook: He at
    ``(0, 0, -d)`` and H at ``(0, 0, +d)`` in PennyLane's default unit (bohr),
    charge +1, STO-3G, Jordan-Wigner, then Z2-symmetry tapering in the
    2-electron sector. The nuclear separation is therefore ``2 d`` bohr; the
    paper's Fig. 7 axis labels ``d`` as an Angstrom separation (LOG.md).

    Parameters
    ----------
    half_distance_bohr : float
        Half the nuclear separation ``d`` in bohr (the reference "distance").

    Returns
    -------
    dict
        ``matrix`` (4x4 complex numpy array, qubit 0 most significant),
        ``fci_energy`` (lowest eigenvalue), ``hf_energy`` (expectation of the
        tapered Hartree-Fock basis state ``|11>``, the paper's plotted
        reference curve), ``separation_bohr`` and ``separation_angstrom``.
    """
    import pennylane as qml

    geometry = np.array(
        [[0.0, 0.0, -half_distance_bohr], [0.0, 0.0, half_distance_bohr]]
    )
    molecule = qml.qchem.Molecule(["He", "H"], geometry, charge=1)
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(molecule)
    generators = qml.symmetry_generators(hamiltonian)
    paulix_ops = qml.paulix_ops(generators, n_qubits)
    sector = qml.qchem.optimal_sector(hamiltonian, generators, 2)
    tapered = qml.taper(hamiltonian, generators, paulix_ops, sector)
    wires = sorted(tapered.wires)
    matrix = np.asarray(qml.matrix(tapered, wire_order=wires), dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "matrix": matrix,
        "fci_energy": float(eigenvalues.min()),
        "hf_energy": float(np.real(matrix[3, 3])),
        "n_qubits_full": int(n_qubits),
        "n_qubits_tapered": len(wires),
        "separation_bohr": 2 * half_distance_bohr,
        "separation_angstrom": 2 * half_distance_bohr * BOHR_TO_ANGSTROM,
    }
