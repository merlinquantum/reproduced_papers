#!/bin/bash
# Run all EGAS reproduction experiments with progress tracking

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
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

# Config directory: set PAPER_SCALE=yes to use the FAITHFUL paper-scale configs in
# configs/paper/ (480-dim 8-layer GPT, 768 candidates, 4000 iters, 10 G/B, 10 pairs).
# These match arXiv:2605.30866 / qDNA-yonsei Generative-QDE and require a GPU — they are
# NOT expected to finish under CPU/qemu. Default uses the reduced-compute configs/.
CFG_DIR="$SCRIPT_DIR/configs"
if [ "${PAPER_SCALE:-no}" = "yes" ]; then
    CFG_DIR="$SCRIPT_DIR/configs/paper"
    echo -e "${YELLOW}[PAPER_SCALE=yes] Using faithful paper configs in $CFG_DIR (GPU-scale; will not finish under qemu).${NC}"
    echo ""
fi

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
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/egas_PW.json" --outdir "$SCRIPT_DIR/outdir/PW"
echo -e "${GREEN}✓ PW results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[5/12]${NC} Running photonic QKSVM on Phishing (PW) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/photonic_PW.json" --outdir "$SCRIPT_DIR/outdir/photonic_PW"
    echo -e "${GREEN}✓ PW photonic results saved${NC}"
fi
echo ""

# ============ WQ Dataset (Wine Quality) ============
echo -e "${YELLOW}[6/12]${NC} Running EGAS search on Wine Quality (WQ) dataset..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/egas_WQ.json" --outdir "$SCRIPT_DIR/outdir/WQ"
echo -e "${GREEN}✓ WQ results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[7/12]${NC} Running photonic QKSVM on Wine Quality (WQ) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/photonic_WQ.json" --outdir "$SCRIPT_DIR/outdir/photonic_WQ"
    echo -e "${GREEN}✓ WQ photonic results saved${NC}"
fi
echo ""

# ============ MGT Dataset (MAGIC Gamma Telescope) ============
echo -e "${YELLOW}[8/12]${NC} Running EGAS search on MAGIC Gamma Telescope (MGT) dataset..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/egas_MGT.json" --outdir "$SCRIPT_DIR/outdir/MGT"
echo -e "${GREEN}✓ MGT results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[9/12]${NC} Running photonic QKSVM on MAGIC Gamma Telescope (MGT)..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/photonic_MGT.json" --outdir "$SCRIPT_DIR/outdir/photonic_MGT"
    echo -e "${GREEN}✓ MGT photonic results saved${NC}"
fi
echo ""

# ============ WDGV1 Dataset (Waveform, multiclass) ============
echo -e "${YELLOW}[10/12]${NC} Running EGAS search on Waveform DB (WDGV1) dataset (multiclass)..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/egas_WDGV1.json" --outdir "$SCRIPT_DIR/outdir/WDGV1"
echo -e "${GREEN}✓ WDGV1 results saved${NC}"

if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo -e "${YELLOW}[11/12]${NC} Running photonic QKSVM on Waveform DB (WDGV1) (multiclass)..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$CFG_DIR/photonic_WDGV1.json" --outdir "$SCRIPT_DIR/outdir/photonic_WDGV1"
    echo -e "${GREEN}✓ WDGV1 photonic results saved${NC}"
fi
echo ""

# Generate all plots
echo -e "${YELLOW}[12/12]${NC} Generating all figures..."
cd "$SCRIPT_DIR"

# Temporarily disable set -e for plot generation (plots are optional)
set +e

# Helper function to find latest metrics by run directory timestamp (YYYYMMDD-HHMMSS)
find_latest_metrics() {
    local dataset_dir="$1"
    if [ ! -d "$dataset_dir" ]; then
        echo ""
        return
    fi

    local latest_run=""
    local latest_mtime=0
    for run_dir in "$dataset_dir"/run_*; do
        if [ -d "$run_dir" ]; then
            local mtime
            if stat --version >/dev/null 2>&1; then
                mtime=$(stat -c "%Y" "$run_dir" 2>/dev/null || echo 0)
            else
                mtime=$(stat -f "%m" "$run_dir" 2>/dev/null || echo 0)
            fi
            if [ "$mtime" -gt "$latest_mtime" ]; then
                latest_mtime="$mtime"
                latest_run="$run_dir"
            fi
        fi
    done

    if [ -n "$latest_run" ] && [ -f "$latest_run/metrics.json" ]; then
        echo "$latest_run/metrics.json"
    else
        echo ""
    fi
}

# Find latest metrics for each dataset
PW_GATE=$(find_latest_metrics "outdir/PW")
WQ_GATE=$(find_latest_metrics "outdir/WQ")
MGT_GATE=$(find_latest_metrics "outdir/MGT")
WDGV1_GATE=$(find_latest_metrics "outdir/WDGV1")

# Diagnostic plots first (Wasserstein and Fig 1)
WASSERSTEIN_METRICS=$(find_latest_metrics "outdir/wasserstein")
FIG1_METRICS=$(find_latest_metrics "outdir/fig1")

if [ -n "$WASSERSTEIN_METRICS" ]; then
    echo "  Generating Table I (Wasserstein)..."
    $PYTHON utils/plot_results.py --wasserstein "$WASSERSTEIN_METRICS"
else
    echo "  ⚠ Wasserstein metrics not found in outdir/wasserstein"
fi

if [ -n "$FIG1_METRICS" ]; then
    echo "  Generating Fig 1 (trace distance)..."
    $PYTHON utils/plot_results.py --fig1 "$FIG1_METRICS"
else
    echo "  ⚠ Fig 1 metrics not found in outdir/fig1"
fi

# Fig 3 (PW only)
if [ -n "$PW_GATE" ]; then
    echo "  Generating Fig 3 (gate)..."
    $PYTHON utils/plot_results.py --fig3-gate "$PW_GATE"
else
    echo "  ⚠ PW gate metrics not found in outdir/PW"
fi

if [ "$RUN_PHOTONIC" = "yes" ]; then
    PW_PHOTONIC=$(find_latest_metrics "outdir/photonic_PW")
    if [ -n "$PW_PHOTONIC" ]; then
        echo "  Generating Fig 3 (photonic)..."
        $PYTHON utils/plot_results.py --fig3-photonic "$PW_PHOTONIC"
    else
        echo "  ⚠ PW photonic metrics not found in outdir/photonic_PW"
    fi
fi

# Fig 4 & 5 (all datasets)
if [ -n "$PW_GATE" ] && [ -n "$WQ_GATE" ] && [ -n "$MGT_GATE" ] && [ -n "$WDGV1_GATE" ]; then
    echo "  Generating Fig 4 (gate, all datasets)..."
    $PYTHON utils/plot_results.py --fig4-gate "$PW_GATE" "$WQ_GATE" "$MGT_GATE" "$WDGV1_GATE"
    echo "  Generating Fig 5 (gate, all datasets)..."
    $PYTHON utils/plot_results.py --fig5-gate "$PW_GATE" "$WQ_GATE" "$MGT_GATE" "$WDGV1_GATE"
else
    echo "  ⚠ Some gate metrics missing: PW=$PW_GATE WQ=$WQ_GATE MGT=$MGT_GATE WDGV1=$WDGV1_GATE"
fi

if [ "$RUN_PHOTONIC" = "yes" ]; then
    PW_PHOTONIC=$(find_latest_metrics "outdir/photonic_PW")
    WQ_PHOTONIC=$(find_latest_metrics "outdir/photonic_WQ")
    MGT_PHOTONIC=$(find_latest_metrics "outdir/photonic_MGT")
    WDGV1_PHOTONIC=$(find_latest_metrics "outdir/photonic_WDGV1")
    if [ -n "$PW_PHOTONIC" ] && [ -n "$WQ_PHOTONIC" ] && [ -n "$MGT_PHOTONIC" ] && [ -n "$WDGV1_PHOTONIC" ]; then
        echo "  Generating Fig 4 (photonic, all datasets)..."
        $PYTHON utils/plot_results.py --fig4-photonic "$PW_PHOTONIC" "$WQ_PHOTONIC" "$MGT_PHOTONIC" "$WDGV1_PHOTONIC"
        echo "  Generating Fig 5 (photonic, all datasets)..."
        $PYTHON utils/plot_results.py --fig5-photonic "$PW_PHOTONIC" "$WQ_PHOTONIC" "$MGT_PHOTONIC" "$WDGV1_PHOTONIC"
    else
        echo "  ⚠ Some photonic metrics missing: PW=$PW_PHOTONIC WQ=$WQ_PHOTONIC MGT=$MGT_PHOTONIC WDGV1=$WDGV1_PHOTONIC"
    fi
fi

# Re-enable set -e
set -e

echo -e "${GREEN}✓ Plot generation complete${NC}"
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
