"""Acquire pinned reference inputs and reconcile candidate source datasets."""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import sha256, write_json
from .data_setup import (
    audit_data,
    load_evaluation_hlas,
    load_pseudosequences,
    load_training_data,
)


def compare_training(reference: Path, candidate: Path) -> dict:
    """Compare ordered rows, pair multiplicities, unknown residues and HLA counts.

    Parameters
    ----------
    reference, candidate : Path
        Headerless peptide-label-HLA files to compare without filtering.

    Returns
    -------
    dict
        Hashes, ordered/multiset equality and aggregate difference counts.
    """
    left, right = load_training_data(reference), load_training_data(candidate)
    a = Counter((r.peptide, r.hla) for r in left)
    b = Counter((r.peptide, r.hla) for r in right)
    counts_a, counts_b = Counter(r.hla for r in left), Counter(r.hla for r in right)
    return {
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "reference_rows": len(left),
        "candidate_rows": len(right),
        "reference_hlas": len(counts_a),
        "candidate_hlas": len(counts_b),
        "pair_multiset_equal": a == b,
        "ordered_rows_equal": left == right,
        "labels_equal_and_positive": all(r.label == 1 for r in left + right),
        "missing_pair_occurrences": sum((a - b).values()),
        "extra_pair_occurrences": sum((b - a).values()),
        "reference_duplicate_occurrences": len(left) - len(a),
        "candidate_duplicate_occurrences": len(right) - len(b),
        "reference_X_rows": sum("X" in r.peptide for r in left),
        "candidate_X_rows": sum("X" in r.peptide for r in right),
        "per_hla": {
            h: {"reference": counts_a[h], "candidate": counts_b[h]}
            for h in sorted(counts_a.keys() | counts_b.keys())
        },
    }


def prepare_reference(
    data_root: Path, project: Path, *, opener=urllib.request.urlopen
) -> dict:
    """Download immutable author inputs, verify bytes, then audit without filtering.

    Parameters
    ----------
    data_root : Path
        External raw-data directory, outside the public paper folder.
    project : Path
        Paper folder containing the pinned source manifest.
    opener : callable
        URL opener; injectable for offline failure-path tests.

    Returns
    -------
    dict
        Acquisition provenance and reference-dataset audit.
    """
    data_root, project = data_root.expanduser().resolve(), project.resolve()
    if data_root == project or project in data_root.parents:
        raise ValueError("Raw datasets must be outside the public paper folder")
    manifest = json.loads((project / "assets/training_data_sources.json").read_text())
    data_root.mkdir(parents=True, exist_ok=True)
    acquired = []
    for entry in manifest["files"]:
        if entry["name"] == "HLA_9mer_test.txt":
            continue  # The held-out test file is not used by this reproduction.
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("Source manifest names must be basenames")
        target = data_root / name
        if target.exists():
            if sha256(target) != entry["sha256"]:
                raise ValueError(f"Refusing to replace changed input: {name}")
        else:
            temporary = target.with_suffix(target.suffix + ".download")
            created = False
            try:
                with temporary.open("xb") as out:
                    created = True
                    with opener(entry["url"], timeout=90) as response:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                if (
                    temporary.stat().st_size != entry["bytes"]
                    or sha256(temporary) != entry["sha256"]
                ):
                    raise ValueError(f"Downloaded input hash/size mismatch: {name}")
                temporary.replace(target)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
        acquired.append(
            {
                "name": name,
                "url": entry["url"],
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
        )
    records = load_training_data(data_root / "HLA_pos9mers.txt")
    hlas = load_evaluation_hlas(data_root / "unique_hla_allotypes.txt")
    mapping = load_pseudosequences(
        data_root / "MHC_pseudo_HLA_only.dat", {r.hla for r in records} | set(hlas)
    )
    audit = audit_data(records, mapping, hlas)
    if (
        audit["training_rows"],
        audit["training_hlas"],
        audit["evaluation_hlas"],
        audit["rows_with_X"],
    ) != (105971, 126, 131, 2):
        raise ValueError("Pinned reference dataset identity failed")
    result = {
        "status": "complete",
        "source": "pinned_author_reference",
        "revision": manifest["revision"],
        "retrieved_or_verified_utc": datetime.now(timezone.utc).isoformat(),
        "transformations": "None; preserve every nonblank row and its order",
        "files": acquired,
        "audit": audit,
        "iedb_substitution": False,
        "rights_note": "User download is not a blanket redistribution grant; see DATA_SOURCES.md",
    }
    write_json(data_root / "data_provenance.json", result)
    return result
