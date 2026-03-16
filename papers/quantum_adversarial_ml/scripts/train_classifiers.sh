#!/bin/bash
# Train quantum and classical classifiers
# Produces Figure 4-5 training curves

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

cd "$PAPER_DIR"

echo "=== Training Classifiers ==="

# Binary quantum classifier (digits 1 vs 9)
echo "Training quantum classifier (binary)..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/train_quantum.json \
    --outdir results/train_quantum

# 4-class quantum classifier
echo ""
echo "Training quantum classifier (4-class)..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/mnist_multi_train.json \
    --outdir results/train_quantum_4class

echo ""
echo "Training curves saved in results/"
