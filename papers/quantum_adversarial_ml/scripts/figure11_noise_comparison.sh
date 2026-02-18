#!/bin/bash
# Reproduce Figure 11: Random noise vs adversarial perturbations
# Shows adversarial perturbations are much more effective than random noise
#
# NOTE: This simplified version runs at a representative epsilon value.
# For full parameter sweeps, use the noise_comparison.json config or
# modify configs and run multiple times.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

cd "$PAPER_DIR"

echo "=== Figure 11: Noise vs Adversarial Comparison ==="
echo "Paper: Quantum Adversarial Machine Learning (Lu et al., 2020)"
echo ""

# Step 1: Train the quantum classifier (if not already trained)
MODEL_PATH="results/train_quantum/model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Step 1: Training quantum classifier..."
    python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
        --config configs/train_quantum.json \
        --outdir results/train_quantum
    
    MODEL_PATH=$(find results/train_quantum -name "model.pt" | head -1)
    ln -sf "$(basename $(dirname $MODEL_PATH))/model.pt" results/train_quantum/model.pt 2>/dev/null || true
else
    echo "Step 1: Skipping training (model exists)"
fi

# Step 2: Run comprehensive noise comparison using the dedicated config
echo ""
echo "Step 2: Running noise comparison experiment..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/noise_comparison.json \
    --outdir results/noise_comparison

echo ""
echo "=== Figure 11 Complete ==="
echo "Results saved in results/noise_comparison/"
echo ""
echo "The experiment compares:"
echo "  - Adversarial perturbations (BIM attack)"
echo "  - Random uniform noise"  
echo "  - Photon loss (quantum-native noise)"
