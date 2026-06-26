# VISITED_URLS.md — Generative Quantum Data Embeddings (EGAS)

## Remote / Internet URLs
| Resource | Local Cache | Purpose | First access |
|---|---|---|---|
| https://arxiv.org/abs/2605.30866v1 | — | Paper abstract page | 2026-06-23 |
| https://arxiv.org/pdf/2605.30866v1 | $REPRO_SCRATCH_DIR/paper.pdf | Primary paper PDF | 2026-06-23 |
| UCI ML Repo via `ucimlrepo` (ids: PW=327, WDGV1=107, DB=602, WineQuality=186, MGT=159, EGSSD=471) | data/generative_quantum_embeddings/ (cached on fetch) | Datasets for E1–E5 | 2026-06-23 |

## Local resources
| Path | Purpose |
|---|---|
| $REPRO_SCRATCH_DIR/paper.pdf | Paper PDF |
| $REPRO_SCRATCH_DIR/paper.txt | Extracted paper text (pdftotext -layout) |
| /reproduced_papers/papers/nn_embedding/ | Existing NQE reproduction (ref [19], baseline patterns) |
| /home/agent/MERLIN_COOKBOOK.md | MerLin patterns for Phase 4 |

## Cached external data
| Resource | Local Cache | Notes |
|---|---|---|
| UCI datasets | data/generative_quantum_embeddings/*.csv | Cached locally on first fetch to avoid re-download |
