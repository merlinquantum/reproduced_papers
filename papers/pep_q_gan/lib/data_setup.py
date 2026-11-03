"""Standard-library-only data loading and verification."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PeptideRecord:
    peptide: str
    label: int
    hla: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_data(path: Path) -> list[PeptideRecord]:
    """The supplied training file has no header; preserve every nonblank row."""
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path.name}:{number}: expected peptide, label, HLA")
            try:
                record = PeptideRecord(parts[0], int(parts[1]), parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}:{number}: invalid label (no header expected)"
                ) from exc
            records.append(record)
    if not records:
        raise ValueError("Training data is empty")
    return records


def load_pseudosequences(
    path: Path, required_hlas: set[str] | None = None
) -> dict[str, str]:
    """Match upstream last-entry-wins outside the requested HLA set; reject relevant conflicts."""
    mapping = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path.name}:{number}: expected HLA and pseudosequence")
        hla, sequence = parts
        if (
            hla in mapping
            and mapping[hla] != sequence
            and (required_hlas is None or hla in required_hlas)
        ):
            raise ValueError(f"Conflicting pseudosequences for {hla}")
        mapping[hla] = sequence
    return mapping


def load_evaluation_hlas(path: Path) -> list[str]:
    hlas = path.read_text(encoding="utf-8").split()
    if not hlas or len(hlas) != len(set(hlas)):
        raise ValueError("Evaluation HLA list is empty or contains duplicates")
    return hlas


def audit_data(
    records: list[PeptideRecord], mapping: dict[str, str], evaluation_hlas: list[str]
) -> dict:
    """Check structure and coverage; preserve upstream X residues and row order."""
    train_hlas = {row.hla for row in records}
    required_hlas = train_hlas | set(evaluation_hlas)
    missing = sorted(required_hlas - mapping.keys())
    if missing:
        raise ValueError(f"Missing HLA pseudosequences: {missing}")
    checks = {
        "all_peptides_length_9": all(len(row.peptide) == 9 for row in records),
        "all_training_labels_positive": all(row.label == 1 for row in records),
        "peptide_alphabet_standard_or_X": all(
            set(row.peptide) <= AMINO_ACIDS | {"X"} for row in records
        ),
        "all_required_pseudosequences_length_34": all(
            len(mapping[hla]) == 34 for hla in required_hlas
        ),
        "pseudosequence_alphabet_standard": all(
            set(mapping[hla]) <= AMINO_ACIDS for hla in required_hlas
        ),
        "no_duplicate_peptide_hla_pairs": len(records)
        == len({(row.peptide, row.hla) for row in records}),
        "all_training_and_evaluation_hlas_mapped": not missing,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Data checks failed: {failures}")
    unknown_rows = [row for row in records if "X" in row.peptide]
    counts = Counter(row.hla for row in records)
    return {
        "training_rows": len(records),
        "unique_peptides": len({row.peptide for row in records}),
        "training_hlas": len(train_hlas),
        "evaluation_hlas": len(evaluation_hlas),
        "unseen_evaluation_hlas": sorted(set(evaluation_hlas) - train_hlas),
        "peptide_length": 9,
        "hla_pseudosequence_length": 34,
        "rows_with_X": len(unknown_rows),
        "X_examples": [row.__dict__ for row in unknown_rows],
        "most_frequent_training_hlas": counts.most_common(5),
        "checks": checks,
        "row_policy": "Preserve all rows; no header skipping or filtering",
        "test_file_used_for_training": False,
    }
