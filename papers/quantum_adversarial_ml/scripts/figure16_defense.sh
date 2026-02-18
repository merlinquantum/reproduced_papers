#!/bin/bash
# Reproduce Figure 16: Adversarial Training Defense
# Shows how adversarial training improves robustness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

cd "$PAPER_DIR"

echo "=== Figure 16: Adversarial Training Defense ==="
echo "Paper: Quantum Adversarial Machine Learning (Lu et al., 2020)"
echo ""

# Run adversarial training
echo "Training with adversarial examples..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/adversarial_training.json \
    --outdir results/defense

echo ""
echo "=== Figure 16 Complete ==="
echo "Results saved in results/defense/"
echo "Output includes training curves showing clean and adversarial accuracy"
