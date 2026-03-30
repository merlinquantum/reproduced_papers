#!/bin/bash
# Reproduce Figure 7: Attack accuracy vs epsilon
# Runs BIM attack at representative epsilon values
#
# NOTE: This simplified version runs at a few key epsilon values.
# For the full sweep as in the paper, modify configs/attack_eval.json
# and run multiple times with different epsilon values.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"

echo "=== Figure 7: Robustness to BIM Attack ==="
echo "Paper: Quantum Adversarial Machine Learning (Lu et al., 2020)"
echo ""

cd "$PAPER_DIR"

# Step 1: Train the quantum classifier (if not already trained)
MODEL_PATH="results/train_quantum/model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Step 1: Training quantum classifier..."
    python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
        --config configs/train_quantum.json \
        --outdir results/train_quantum
    
    # Find the actual model.pt path (may be in a timestamped subdirectory)
    MODEL_PATH=$(find results/train_quantum -name "model.pt" | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Model training did not produce model.pt"
        exit 1
    fi
    # Create symlink for convenience
    ln -sf "$(basename $(dirname $MODEL_PATH))/model.pt" results/train_quantum/model.pt 2>/dev/null || true
else
    echo "Step 1: Skipping training (model exists at $MODEL_PATH)"
fi

# Step 2: Run BIM attack at eps=0.05 (representative value)
echo ""
echo "Step 2: Running BIM attack at epsilon=0.05..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/attack_bim_eps005.json \
    --outdir results/attack_bim_eps005

# Step 3: Run BIM attack at eps=0.10
echo ""
echo "Step 3: Running BIM attack at epsilon=0.10..."
python "$REPO_ROOT/implementation.py" --paper quantum_adversarial_ml \
    --config configs/attack_bim_eps010.json \
    --outdir results/attack_bim_eps010

echo ""
echo "=== Figure 7 Complete ==="
echo "Results saved in results/attack_bim_eps*/"
echo ""
echo "To run at additional epsilon values, modify attack.epsilon in"
echo "configs/attack_eval.json and run again."
