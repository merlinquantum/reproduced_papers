#!/bin/bash
# Reproduce all figures and tables from Lu et al. (2020)
# "Quantum Adversarial Machine Learning"
#
# Plus: Extended comparison of amplitude vs angle encoding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "Quantum Adversarial Machine Learning"
echo "Lu et al. (2020) - Full Paper Reproduction"
echo "================================================"
echo ""

# Create output directory
mkdir -p "$PAPER_DIR/results"
cd "$PAPER_DIR"

# Figure 4-5: Training curves
echo "[1/7] Figures 4-5: Training MNIST classifiers..."
bash "$SCRIPT_DIR/train_classifiers.sh"

# Figure 7: Robustness sweep
echo ""
echo "[2/7] Figure 7: Robustness to BIM attack..."
bash "$SCRIPT_DIR/figure7_robustness.sh"

# Figure 11: Noise comparison
echo ""
echo "[3/7] Figure 11: Noise vs adversarial comparison..."
bash "$SCRIPT_DIR/figure11_noise_comparison.sh"

# Table III: Transfer attacks
echo ""
echo "[4/7] Table III: Black-box transfer attacks..."
bash "$SCRIPT_DIR/table3_transfer.sh"

# Figure 16: Adversarial training
echo ""
echo "[5/7] Figure 16: Adversarial training defense..."
bash "$SCRIPT_DIR/figure16_defense.sh"

# Topological phases (Section III.C)
echo ""
echo "[6/7] Section III.C: Topological phase classification..."
bash "$SCRIPT_DIR/topological_phases.sh"

# Extended: Amplitude vs Angle Encoding Comparison
echo ""
echo "[7/7] Extended: Amplitude vs Angle Encoding Comparison..."
bash "$SCRIPT_DIR/encoding_comparison.sh"

echo ""
echo "================================================"
echo "All experiments complete!"
echo "================================================"
echo ""
echo "Results saved in $PAPER_DIR/results/"
echo ""
echo "Paper reproduction (Lu et al. 2020):"
echo "  - results/train_quantum/       : Amplitude-encoded classifier"
echo "  - results/attack_bim_eps*/     : BIM attack results"
echo "  - results/noise_comparison/    : Noise vs adversarial (Fig 11)"
echo "  - results/transfer_attacks/    : Black-box transfer (Table III)"
echo "  - results/defense/             : Adversarial training (Fig 16)"
echo "  - results/topological/         : Topological phases (Sec III.C)"
echo ""
echo "Extended analysis:"
echo "  - results/train_angle/         : Angle-encoded classifier"
echo "  - results/attack_angle_eps*/   : Attacks on angle-encoded model"
echo "  - results/noise_angle/         : Noise comparison for angle encoding"
echo ""
echo "Compare amplitude vs angle encoding results to see if"
echo "encoding method affects adversarial robustness!"
echo "================================================"
