#!/bin/bash
# Run the Fourier Fingerprints reproduction.
#
# Usage:
#   bash run.sh
#
# Results are written to the output directory defined in configs/default.json.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running Fourier Fingerprints"
echo "=========================================="

configs=(
	"default.json"
	"1D_exp.json"
	"2D_linear.json"
	"2D_exp.json"
	"2D_balanced.json"
)

for config in "${configs[@]}"; do
	echo ""
	echo "--- Fourier Fingerprint: $config ---"
	python "$SCRIPT_DIR/implementation.py" \
		--config "$SCRIPT_DIR/configs/$config"
done


