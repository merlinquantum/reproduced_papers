#!/bin/bash
# Run the Fourier Fingerprints reproduction through the repository-wide runner.
#
# Usage:
#   bash run.sh
#
# Results are written to the output directory defined in each config.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Running Fourier Fingerprints"
echo "=========================================="

configs=(
	"defaults.json"
	"1D_exp.json"
	"1D_balanced.json"
	"2D_linear.json"
	"2D_exp.json"
	"2D_balanced.json"
)

for config in "${configs[@]}"; do
	echo ""
	echo "--- Fourier Fingerprint: $config ---"
	python "$REPO_ROOT/implementation.py" \
		--paper Fourier_Fingerprints \
		--config "configs/$config"
done
