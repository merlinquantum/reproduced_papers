"""DataS1 populations, experimental assay truth and residue bit encoding.

DataS1 is the supplementary data file of Peters et al., Science Advances (2026),
DOI 10.1126/sciadv.aec3824. It is downloaded once into
``<data_root>/QCHIPP/paper_supplement/`` and verified by SHA-256 on every load.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import torch

from runtime_lib.data_paths import paper_data_dir

LOGGER = logging.getLogger(__name__)

PAPER_NAME = "QCHIPP"
SUPPLEMENT_DIRNAME = "paper_supplement"
DATASET_FILENAME = "aec3824_data_file_s1.csv"
DATASET_ARCHIVE_NAME = "sciadv.aec3824_data_file_s1.zip"
DATASET_SHA256 = "2b91dde1046d46f9ec0bd64ae1d2bc0d148da09b065287aef44a1238c0bb3b07"
# Europe PMC serves every supplement of PMC13398535 as one zip. The PMC article
# page answers scripted downloads with a bot challenge instead of the file.
SUPPLEMENT_BUNDLE_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13398535/supplementaryFiles"
)
DOWNLOAD_TIMEOUT_SECONDS = 120

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
TASK_COLUMNS = {
    "binding": ("Binding_split", "MHC.Assay"),
    "immunogenic": ("Immunogenic_split", "T.cell.response"),
    "mixed": ("Immunogenic_mix_split", "T.cell.response"),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download_dataset_csv() -> bytes:
    """Fetch the Europe PMC supplement bundle and extract the DataS1 CSV bytes."""
    LOGGER.info("Downloading Q-CHIPP DataS1 from %s", SUPPLEMENT_BUNDLE_URL)
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https URL
            SUPPLEMENT_BUNDLE_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            bundle_bytes = response.read()
    except OSError as error:
        raise RuntimeError(
            f"Could not download the Q-CHIPP supplements from {SUPPLEMENT_BUNDLE_URL}. "
            f"Download them manually and place {DATASET_FILENAME} under "
            f"data/{PAPER_NAME}/{SUPPLEMENT_DIRNAME}/ (see README 'Data')."
        ) from error
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
        archive_bytes = bundle.read(DATASET_ARCHIVE_NAME)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return archive.read(DATASET_FILENAME)


def ensure_dataset(data_root: str | Path | None) -> Path:
    """Return the verified DataS1 CSV path, downloading it on first use.

    Parameters
    ----------
    data_root : str | Path | None
        Shared data root. ``None`` follows the runtime convention
        (``DATA_DIR`` environment variable, then repository ``data/``).

    Returns
    -------
    Path
        Path to ``aec3824_data_file_s1.csv``.

    Raises
    ------
    RuntimeError
        If the file is absent and cannot be downloaded.
    ValueError
        If the local or downloaded file does not match the published SHA-256.
    """
    supplement_dir = (
        paper_data_dir(PAPER_NAME, data_root=data_root) / SUPPLEMENT_DIRNAME
    )
    csv_path = supplement_dir / DATASET_FILENAME
    if csv_path.exists():
        observed = _sha256_bytes(csv_path.read_bytes())
        if observed != DATASET_SHA256:
            raise ValueError(
                f"{csv_path} has SHA-256 {observed}, expected {DATASET_SHA256}. "
                "Delete it to re-download the published DataS1."
            )
        return csv_path

    csv_bytes = _download_dataset_csv()
    observed = _sha256_bytes(csv_bytes)
    if observed != DATASET_SHA256:
        raise ValueError(
            f"Downloaded DataS1 has SHA-256 {observed}, expected {DATASET_SHA256}."
        )
    supplement_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(".csv.tmp")
    temporary_path.write_bytes(csv_bytes)
    temporary_path.replace(csv_path)
    LOGGER.info("Saved verified DataS1 to %s", csv_path)
    return csv_path


def read_rows(path: Path) -> list[dict]:
    """Read every DataS1 row as a dict keyed by column name."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def task_rows(rows: list[dict], task: str) -> tuple[list[dict], list[str]]:
    """Select one task's released train/test rows with experimental labels.

    Parameters
    ----------
    rows : list of dict
        DataS1 rows from :func:`read_rows`.
    task : str
        ``binding``, ``immunogenic`` or ``mixed``.

    Returns
    -------
    tuple
        The selected rows, and the peptides excluded for missing assay truth.

    Raises
    ------
    ValueError
        If a selected row breaks the population assumptions (canonical
        nine-mer, HLA-A02:01, confirmed binder for ``immunogenic``).
    """
    split_column, label_column = TASK_COLUMNS[task]
    released = [r for r in rows if r[split_column] in ("train", "test")]
    # Assay truth only: rows without an experimental label are excluded and
    # reported, never filled from predicted affinity or derived annotations.
    missing = [
        r["Mutant_peptide"]
        for r in released
        if r[label_column] not in ("Positive", "Negative")
    ]
    selected = [r for r in released if r[label_column] in ("Positive", "Negative")]
    for row in selected:
        peptide = row["Mutant_peptide"]
        if len(peptide) != 9 or set(peptide) - set(ALPHABET):
            raise ValueError(f"Noncanonical peptide in released split: {peptide}")
        if row["HLA_allele"] != "HLA-A02:01":
            raise ValueError("Unexpected allele")
        if task == "immunogenic" and row["MHC.Assay"] != "Positive":
            raise ValueError("Binder-only population contains a nonconfirmed binder")
    return selected, missing


def load_task(cfg: dict) -> dict:
    """Load one released task split with experimental labels and bit encoding.

    Parameters
    ----------
    cfg : dict
        Needs ``dataset.task`` (``binding``, ``immunogenic`` or ``mixed``) and
        ``dataset.positions`` (four distinct one-based residue positions).
        ``data_root`` is optional and follows the shared runtime convention.

    Returns
    -------
    dict
        ``train`` and ``test`` entries with ``bits`` (B, 20), ``targets`` (B,)
        and ``peptides``, plus the source hash, the category codes and the
        peptides excluded for missing assay truth.

    Raises
    ------
    ValueError
        If positions are invalid or the released split violates the
        population assumptions (allele, canonical residues, binder-only
        immunogenicity, train/test overlap).
    """
    settings = cfg["dataset"]
    task = settings["task"]
    split_column, label_column = TASK_COLUMNS[task]
    positions = settings["positions"]
    if (
        len(positions) != 4
        or len(set(positions)) != 4
        or any(p not in range(1, 10) for p in positions)
    ):
        raise ValueError("Select four distinct one-based residue positions in 1..9")
    path = ensure_dataset(cfg.get("data_root"))
    selected, missing = task_rows(read_rows(path), task)
    # The source BinaryEncoder fits sorted observed categories across train and
    # test. The release contains all 20 residues, so this transductive vocabulary
    # is disclosed rather than hidden.
    vocabulary = sorted(
        {
            ALPHABET.index(r["Mutant_peptide"][p - 1])
            for r in selected
            for p in positions
        }
    )
    category_codes = {ALPHABET[a]: i + 1 for i, a in enumerate(vocabulary)}
    result = {
        "source_path": str(path),
        "source_sha256": DATASET_SHA256,
        "excluded_missing_assay": missing,
        "category_codes": category_codes,
        "encoding": (
            "source sorted-union category ranks 1..20, five-bit MSB-first; "
            "transductive vocabulary"
        ),
    }
    for split in ("train", "test"):
        subset = [r for r in selected if r[split_column] == split]
        bits = [
            encode_peptide(r["Mutant_peptide"], positions, category_codes)
            for r in subset
        ]
        result[split] = {
            "bits": torch.tensor(bits, dtype=torch.float32),
            "targets": torch.tensor(
                [r[label_column] == "Positive" for r in subset], dtype=torch.float32
            ),
            "peptides": [r["Mutant_peptide"] for r in subset],
        }
    if set(result["train"]["peptides"]) & set(result["test"]["peptides"]):
        raise ValueError("Released train/test peptide overlap")
    return result


def encode_peptide(
    peptide: str, positions: list[int], category_codes: dict[str, int]
) -> list[int]:
    """Encode four selected residues as 4 x 5 MSB-first category-rank bits.

    Parameters
    ----------
    peptide : str
        Canonical nine-residue peptide.
    positions : list of int
        Four one-based residue positions.
    category_codes : dict
        Residue letter to category rank (1..20) fitted by :func:`load_task`.

    Returns
    -------
    list of int
        Twenty bits, five per selected residue.
    """
    return [
        (category_codes[peptide[p - 1]] >> shift) & 1
        for p in positions
        for shift in range(4, -1, -1)
    ]


def one_hot_features(bits: torch.Tensor) -> torch.Tensor:
    """Decode four category-rank codes into the equivalent 80-column one-hot input.

    Parameters
    ----------
    bits : torch.Tensor
        Binary residue bits of shape (B, 20).

    Returns
    -------
    torch.Tensor
        One-hot features of shape (B, 80), 20 columns per residue.
    """
    if (
        bits.ndim != 2
        or bits.shape[1] != 20
        or not torch.all((bits == 0) | (bits == 1))
    ):
        raise ValueError("Expected binary source residue bits with shape (B,20)")
    codes = (
        bits.reshape(-1, 4, 5).long() * bits.new_tensor([16, 8, 4, 2, 1]).long()
    ).sum(-1) - 1
    if torch.any((codes < 0) | (codes >= 20)):
        raise ValueError("Residue code outside the canonical 20-category alphabet")
    return (
        torch.nn.functional.one_hot(codes, num_classes=20).flatten(1).to(torch.float32)
    )
