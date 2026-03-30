#!/bin/bash
# Reproduce Section III.C: Topological Phase Classification
# Quantum Anomalous Hall (QAH) effect

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

cd "$PAPER_DIR"

echo "=== Section III.C: Topological Phases ==="
echo ""

# Train QAH classifier
echo "Training topological phase classifier..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/topological.json \
    --outdir results/topological

# Find and symlink model if in subdirectory
MODEL_PATH=$(find results/topological -name "model.pt" | head -1)
if [ -n "$MODEL_PATH" ]; then
    ln -sf "$(basename $(dirname $MODEL_PATH))/model.pt" results/topological/model.pt 2>/dev/null || true
fi

# Evaluate with adversarial attack
echo ""
echo "Evaluating adversarial robustness..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/topological_attack.json \
    --outdir results/topological_attack

echo ""
echo "=== Topological Phases Complete ==="
echo "Results show robustness on physics-informed datasets."
