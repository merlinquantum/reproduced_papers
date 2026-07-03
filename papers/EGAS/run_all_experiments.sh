#!/bin/bash
# Run all EGAS reproduction experiments with progress tracking

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON="/Users/lfvigneux/miniconda3/envs/reproduce/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if repo root is correct
if [ ! -f "$REPO_ROOT/implementation.py" ]; then
    echo "Error: Could not find implementation.py at repo root: $REPO_ROOT"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EGAS Reproduction - All Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track which experiments to run
RUN_TESTS=${1:-"yes"}
RUN_GATE_EGAS=${2:-"yes"}
RUN_PHOTONIC=${3:-"yes"}

# Test suite
if [ "$RUN_TESTS" = "yes" ]; then
    echo -e "${YELLOW}[1/8]${NC} Running photonic implementation tests..."
    cd "$SCRIPT_DIR"
    $PYTHON -m pytest tests/test_photonic_impl.py -v --tb=short
    echo -e "${GREEN}✓ Tests passed${NC}"
    echo ""
fi

# Wasserstein (Table I)
echo -e "${YELLOW}[2/8]${NC} Running Wasserstein diagnostic (Table I)..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/wasserstein.json" --outdir "$SCRIPT_DIR/outdir/wasserstein"
echo -e "${GREEN}✓ Wasserstein results saved${NC}"
echo ""

# Fig 1
echo -e "${YELLOW}[3/16]${NC} Running Fig 1 experiments (trace distance vs W1)..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/fig1.json" --outdir "$SCRIPT_DIR/outdir/fig1"
echo -e "${GREEN}✓ Fig 1 results saved${NC}"
echo ""

# ============ PW Dataset (Phishing) ============
echo -e "${YELLOW}[4/12]${NC} Running EGAS search on Phishing (PW) dataset..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_PW.json" --outdir "$SCRIPT_DIR/outdir/PW"
echo -e "${GREEN}✓ PW results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[5/12]${NC} Running photonic QKSVM on Phishing (PW) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/photonic_PW.json" --outdir "$SCRIPT_DIR/outdir/photonic_PW"
    echo -e "${GREEN}✓ PW photonic results saved${NC}"
fi
echo ""

# ============ WQ Dataset (Wine Quality) ============
echo -e "${YELLOW}[6/12]${NC} Running EGAS search on Wine Quality (WQ) dataset..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_WQ.json" --outdir "$SCRIPT_DIR/outdir/WQ"
echo -e "${GREEN}✓ WQ results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[7/12]${NC} Running photonic QKSVM on Wine Quality (WQ) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/photonic_WQ.json" --outdir "$SCRIPT_DIR/outdir/photonic_WQ"
    echo -e "${GREEN}✓ WQ photonic results saved${NC}"
fi
echo ""

# ============ MGT Dataset (MAGIC Gamma Telescope) ============
echo -e "${YELLOW}[8/12]${NC} Running EGAS search on MAGIC Gamma Telescope (MGT) dataset..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_MGT.json" --outdir "$SCRIPT_DIR/outdir/MGT"
echo -e "${GREEN}✓ MGT results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[9/12]${NC} Running photonic QKSVM on MAGIC Gamma Telescope (MGT)..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/photonic_MGT.json" --outdir "$SCRIPT_DIR/outdir/photonic_MGT"
    echo -e "${GREEN}✓ MGT photonic results saved${NC}"
fi
echo ""

# ============ WDGV1 Dataset (Waveform, multiclass) ============
echo -e "${YELLOW}[10/12]${NC} Running EGAS search on Waveform DB (WDGV1) dataset (multiclass)..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_WDGV1.json" --outdir "$SCRIPT_DIR/outdir/WDGV1"
echo -e "${GREEN}✓ WDGV1 results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[11/12]${NC} Running photonic QKSVM on Waveform DB (WDGV1) (multiclass)..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/photonic_WDGV1.json" --outdir "$SCRIPT_DIR/outdir/photonic_WDGV1"
    echo -e "${GREEN}✓ WDGV1 photonic results saved${NC}"
fi
echo ""

# Generate all plots
echo -e "${YELLOW}[12/12]${NC} Generating all figures..."
cd "$SCRIPT_DIR"

# Find latest metrics for each dataset
PW_GATE=$(find outdir/PW -name metrics.json -type f 2>/dev/null | sort -r | head -1)
WQ_GATE=$(find outdir/WQ -name metrics.json -type f 2>/dev/null | sort -r | head -1)
MGT_GATE=$(find outdir/MGT -name metrics.json -type f 2>/dev/null | sort -r | head -1)
WDGV1_GATE=$(find outdir/WDGV1 -name metrics.json -type f 2>/dev/null | sort -r | head -1)

# Fig 3 (PW only)
if [ -n "$PW_GATE" ]; then
    echo "  Generating Fig 3 (gate)..."
    $PYTHON utils/plot_results.py --fig3-gate "$PW_GATE"
fi
if [ "$RUN_PHOTONIC" = "yes" ]; then
    PW_PHOTONIC=$(find outdir/photonic_PW -name metrics.json -type f 2>/dev/null | sort -r | head -1)
    if [ -n "$PW_PHOTONIC" ]; then
        echo "  Generating Fig 3 (photonic)..."
        $PYTHON utils/plot_results.py --fig3-photonic "$PW_PHOTONIC"
    fi
fi

# Fig 4 & 5 (all datasets)
if [ -n "$PW_GATE" ] && [ -n "$WQ_GATE" ] && [ -n "$MGT_GATE" ] && [ -n "$WDGV1_GATE" ]; then
    echo "  Generating Fig 4 & 5 (gate, all datasets)..."
    $PYTHON utils/plot_results.py --fig4-gate "$PW_GATE" "$WQ_GATE" "$MGT_GATE" "$WDGV1_GATE"
    $PYTHON utils/plot_results.py --fig5-gate "$PW_GATE" "$WQ_GATE" "$MGT_GATE" "$WDGV1_GATE"
fi

if [ "$RUN_PHOTONIC" = "yes" ]; then
    PW_PHOTONIC=$(find outdir/photonic_PW -name metrics.json -type f 2>/dev/null | sort -r | head -1)
    WQ_PHOTONIC=$(find outdir/photonic_WQ -name metrics.json -type f 2>/dev/null | sort -r | head -1)
    MGT_PHOTONIC=$(find outdir/photonic_MGT -name metrics.json -type f 2>/dev/null | sort -r | head -1)
    WDGV1_PHOTONIC=$(find outdir/photonic_WDGV1 -name metrics.json -type f 2>/dev/null | sort -r | head -1)
    if [ -n "$PW_PHOTONIC" ] && [ -n "$WQ_PHOTONIC" ] && [ -n "$MGT_PHOTONIC" ] && [ -n "$WDGV1_PHOTONIC" ]; then
        echo "  Generating Fig 4 & 5 (photonic, all datasets)..."
        $PYTHON utils/plot_results.py --fig4-photonic "$PW_PHOTONIC" "$WQ_PHOTONIC" "$MGT_PHOTONIC" "$WDGV1_PHOTONIC"
        $PYTHON utils/plot_results.py --fig5-photonic "$PW_PHOTONIC" "$WQ_PHOTONIC" "$MGT_PHOTONIC" "$WDGV1_PHOTONIC"
    fi
fi

# Diagnostic plots (Wasserstein and Fig 1)
WASSERSTEIN_METRICS=$(find outdir/wasserstein -name metrics.json -type f 2>/dev/null | sort -r | head -1)
FIG1_METRICS=$(find outdir/fig1 -name metrics.json -type f 2>/dev/null | sort -r | head -1)
if [ -n "$WASSERSTEIN_METRICS" ]; then
    echo "  Generating diagnostic plot (Table I)..."
    $PYTHON utils/plot_results.py --wasserstein "$WASSERSTEIN_METRICS"
fi
if [ -n "$FIG1_METRICS" ]; then
    echo "  Generating Fig 1..."
    $PYTHON utils/plot_results.py --fig1 "$FIG1_METRICS"
fi

echo -e "${GREEN}✓ All plots generated${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results and plots saved to:"
echo "  • $SCRIPT_DIR/results/ - All generated figures"
echo ""
echo "Per-dataset results:"
echo "  • Gate-based EGAS:"
echo "    - PW:    $SCRIPT_DIR/outdir/PW/"
echo "    - WQ:    $SCRIPT_DIR/outdir/WQ/"
echo "    - MGT:   $SCRIPT_DIR/outdir/MGT/"
echo "    - WDGV1: $SCRIPT_DIR/outdir/WDGV1/"
if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo "  • Photonic QKSVM:"
    echo "    - PW:    $SCRIPT_DIR/outdir/photonic_PW/"
    echo "    - WQ:    $SCRIPT_DIR/outdir/photonic_WQ/"
    echo "    - MGT:   $SCRIPT_DIR/outdir/photonic_MGT/"
    echo "    - WDGV1: $SCRIPT_DIR/outdir/photonic_WDGV1/"
fi
echo ""
echo "Generated figures:"
echo "  • Table 1: table1_wasserstein.png (Wasserstein diagnostic)"
echo "  • Fig 1: fig1_tracedist_vs_w1.png (trace distance vs W1)"
echo "  • Fig 3: fig3_deltaE_per_candidate.png (gate), fig3_deltaE_per_candidate_photonic.png (photonic)"
echo "  • Fig 4: fig4_deltaE_groups.png (gate), fig4_deltaE_groups_photonic.png (photonic) - Per dataset"
echo "  • Fig 5: fig5_win_tie_loss.png (gate), fig5_win_tie_loss_photonic.png (photonic) - Per dataset"
echo ""
echo "Usage:"
echo "  ./run_all_experiments.sh               # Run all (default)"
echo "  ./run_all_experiments.sh no yes no    # Skip tests, run gate EGAS only"
echo "  ./run_all_experiments.sh yes no yes   # Skip gate EGAS, run tests & photonic"
