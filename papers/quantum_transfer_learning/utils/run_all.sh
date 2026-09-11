#!/bin/bash
# Run all quantum transfer learning experiments
#
# Usage (from repo root):
#   ./papers/quantum_transfer_learning/utils/run_all.sh
#
# Or from paper directory:
#   cd papers/quantum_transfer_learning
#   ./utils/run_all.sh
#
# Run specific experiment:
#   ./utils/run_all.sh spiral
#   ./utils/run_all.sh hymenoptera

set -e

# Find paper directory (works whether run from repo root or paper dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
PAPER_NAME="$(basename "$PAPER_DIR")"

# Find repo root by looking for implementation.py
REPO_ROOT="$PAPER_DIR"
while [[ ! -f "$REPO_ROOT/implementation.py" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

if [[ ! -f "$REPO_ROOT/implementation.py" ]]; then
    echo "Error: Cannot find implementation.py in parent directories"
    exit 1
fi

# Get relative path from repo root to paper dir
PAPER_REL_PATH="${PAPER_DIR#$REPO_ROOT/}"

# Define experiments (name:config pairs)
declare -a EXPERIMENTS=(
    "spiral:configs/spiral.json"
    "hymenoptera:configs/hymenoptera.json"
    "cifar_dogs_cats:configs/cifar_dogs_cats.json"
    "cifar_planes_cars:configs/cifar_planes_cars.json"
)

# Optional filter
FILTER="${1:-}"

echo "========================================"
echo "Quantum Transfer Learning - Run All"
echo "========================================"
echo "Repo root:   $REPO_ROOT"
echo "Paper path:  $PAPER_REL_PATH"
echo ""

cd "$REPO_ROOT"

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    config="${entry##*:}"

    # Skip if filter provided and doesn't match
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        continue
    fi

    # Build full config path relative to repo root
    full_config="$PAPER_REL_PATH/$config"

    echo "----------------------------------------"
    echo "Running: $name"
    echo "Config:  $full_config"
    echo "----------------------------------------"

    python implementation.py \
        --paper "$PAPER_REL_PATH" \
        --config "$full_config"

    echo ""
    echo "✓ $name complete"
    echo ""
done

echo "========================================"
echo "All requested experiments complete!"
echo "========================================"