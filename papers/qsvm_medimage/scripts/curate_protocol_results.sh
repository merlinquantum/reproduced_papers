#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-outdir/qsvm-protocol-matrix}"
CURATED_ROOT="${CURATED_ROOT:-results/protocol_matrix_n500_q4_q6}"

ARTIFACTS=(
  protocol_results_per_seed.csv
  protocol_summary.csv
  protocol_summary.md
)

for artifact in "${ARTIFACTS[@]}"; do
  if [[ ! -f "$SOURCE_ROOT/$artifact" ]]; then
    echo "Missing required artifact: $SOURCE_ROOT/$artifact" >&2
    echo "Run aggregate_protocol_matrix.py for this result root first." >&2
    exit 1
  fi
done

mkdir -p "$CURATED_ROOT"

for artifact in "${ARTIFACTS[@]}"; do
  cp "$SOURCE_ROOT/$artifact" "$CURATED_ROOT/$artifact"
  if ! cmp -s "$SOURCE_ROOT/$artifact" "$CURATED_ROOT/$artifact"; then
    echo "Copied artifact differs from its source: $artifact" >&2
    exit 1
  fi
  echo "Curated: $CURATED_ROOT/$artifact"
done

echo "RUN.md is not generated; review or create it separately for run provenance."
