"""Paper values from the journal Source Data (Nat. Commun. 12, 2631, MOESM2).

The workbook was converted to one CSV per sheet under
``<data_root>/power_of_data_qml/sources/natcommun_esm/source_data_csv/``
(see VISITED_URLS.md). Each loader returns rows keyed like this
reproduction's summary rows: ``(component, quantity, embedding, n, train_size)``.

Journal figure numbering: Fig. 3 = arXiv Fig. 1, Fig. 4 = arXiv Fig. 2,
Supp. Fig. 2-4 = arXiv Figs. S1-S3.
"""

from __future__ import annotations

import csv
from pathlib import Path

SOURCE_DATA_SUBDIR = (
    Path("power_of_data_qml") / "sources" / "natcommun_esm" / "source_data_csv"
)
EMBEDDINGS = ("E1", "E2", "E3")
# The Supp. Fig. 2 sheets contain rows at n=29-30 that are inconsistent with the
# published figure (which stops at n=28); LOG.md records them as spurious.
SPURIOUS_ENGINEERED_QK_SIZES = {29, 30}


def source_data_dir(data_root: str | Path) -> Path:
    """Directory holding the Source Data CSVs."""
    return Path(data_root) / SOURCE_DATA_SUBDIR


def _read_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The journal Source Data CSVs are committed under "
            "data/power_of_data_qml/sources/natcommun_esm/source_data_csv/; see the Data "
            "section of papers/power_of_data_qml/README.md"
        )
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    # Column 0 is the pandas index written by the journal; drop it.
    return [row[1:] for row in rows]


def _value(cell: str) -> float | None:
    return float(cell) if cell.strip() else None


def load_paper_values(data_root: str | Path) -> dict[tuple, float]:
    """Return every Source Data value keyed by summary-row identity.

    Keys are ``(component, quantity, embedding, n, train_size)``; ``embedding``
    or ``train_size`` is ``None`` where it does not apply. Regression values are
    MAE, Dataset (C) values are classification error, engineered values are
    accuracy.
    """
    directory = source_data_dir(data_root)
    values: dict[tuple, float] = {}

    def put(component, quantity, embedding, n, train_size, cell):
        value = _value(cell)
        if value is not None:
            values[(component, quantity, embedding, int(n), train_size)] = value

    # Fig. 3a: columns are Q(E1), Q(E2), Q(E3), PQ(E1), PQ(E2), PQ(E3) by position
    # (the geometric sheet labels its third column "Q (E2)"; AMB-12).
    for sheet, quantity in (
        ("Fig._3_a_dimension.csv", "d"),
        ("Fig._3_a_geometric.csv", "g"),
    ):
        for row in _read_rows(directory / sheet)[1:]:
            n = row[0]
            for position, embedding in enumerate(EMBEDDINGS):
                put(
                    "dimension_geometry",
                    f"{quantity}_quantum_kernel",
                    embedding,
                    n,
                    None,
                    row[1 + position],
                )
                put(
                    "dimension_geometry",
                    f"{quantity}_projected_quantum_kernel",
                    embedding,
                    n,
                    None,
                    row[4 + position],
                )

    # Fig. 3b quantum datasets (MAE at N=600): QK, best classical, PQK.
    for embedding in EMBEDDINGS:
        for row in _read_rows(directory / f"Fig._3_b_Q_{embedding}.csv")[1:]:
            n = row[0]
            put("quantum_dataset", "quantum_kernel", embedding, n, 600, row[1])
            put("quantum_dataset", "best_classical", embedding, n, 600, row[2])
            put(
                "quantum_dataset", "projected_quantum_kernel", embedding, n, 600, row[3]
            )

    # Fig. 3b Dataset (C) (classification error at N=600).
    for row in _read_rows(directory / "Fig._3_b_C.csv")[1:]:
        n = row[0]
        for position, embedding in enumerate(EMBEDDINGS):
            put(
                "classical_dataset",
                f"quantum_kernel_{embedding}",
                None,
                n,
                600,
                row[1 + position],
            )
            put(
                "classical_dataset",
                f"projected_quantum_kernel_{embedding}",
                None,
                n,
                600,
                row[4 + position],
            )
        put("classical_dataset", "best_classical", None, n, 600, row[7])

    # Fig. 4 and Supp. Fig. 2: accuracy; columns C100, C600, PQ100, PQ600, Q100, Q600.
    for component, prefix in (
        ("engineered_pqk", "Fig._4_PQK"),
        ("engineered_qk", "Supp._Fig._2_QK"),
    ):
        for embedding in EMBEDDINGS:
            for row in _read_rows(directory / f"{prefix}_{embedding}_.csv")[1:]:
                n = int(row[0])
                if component == "engineered_qk" and n in SPURIOUS_ENGINEERED_QK_SIZES:
                    continue
                for offset, quantity in (
                    (1, "best_classical"),
                    (3, "projected_quantum_kernel"),
                    (5, "quantum_kernel"),
                ):
                    put(
                        component,
                        quantity,
                        embedding,
                        n,
                        100,
                        row[offset] if offset < len(row) else "",
                    )
                    put(
                        component,
                        quantity,
                        embedding,
                        n,
                        600,
                        row[offset + 1] if offset + 1 < len(row) else "",
                    )

    # Supp. Fig. 4: bound and best-classical error, n = 3..12.
    for sheet, quantity in (
        ("Supp._Fig._4_bound_.csv", "prediction_error_bound"),
        ("Supp._Fig._4_error_.csv", "best_classical"),
    ):
        for row in _read_rows(directory / sheet)[1:]:
            for position, embedding in enumerate(EMBEDDINGS):
                put("bound", quantity, embedding, row[0], 600, row[1 + position])
    return values


def load_training_size_values(data_root: str | Path) -> list[dict]:
    """Per-seed Supp. Fig. 3 rows (Dataset (Q, E2), MAE vs training size)."""
    rows = _read_rows(source_data_dir(data_root) / "Supp._Fig._3.csv")
    header, body = rows[0], rows[1:]
    columns = {name: index for index, name in enumerate(header)}
    return [
        {
            "n": int(row[columns["n"]]),
            "seed": int(row[columns["seed"]]),
            "train_size": int(row[columns["training size"]]),
            "projected_quantum_kernel": float(row[columns["PQK (QNN)"]]),
            "quantum_kernel": float(row[columns["QK (QNN)"]]),
            "best_classical": float(row[columns["Best Classical ML (QNN)"]]),
        }
        for row in body
    ]


__all__ = ["load_paper_values", "load_training_size_values", "source_data_dir"]
