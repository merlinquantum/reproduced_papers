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
echo -e "${YELLOW}[3/8]${NC} Running Fig 1 experiments (trace distance vs W1)..."
cd "$REPO_ROOT"
$PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/fig1.json" --outdir "$SCRIPT_DIR/outdir/fig1"
echo -e "${GREEN}✓ Fig 1 results saved${NC}"
echo ""

if [ "$RUN_GATE_EGAS" = "yes" ]; then
    # EGAS on PW dataset
    echo -e "${YELLOW}[4/8]${NC} Running EGAS search on Phishing (PW) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_PW.json" --outdir "$SCRIPT_DIR/outdir/PW"
    echo -e "${GREEN}✓ PW results saved${NC}"
    echo ""

    # EGAS on WQ dataset
    echo -e "${YELLOW}[5/8]${NC} Running EGAS search on Wine Quality (WQ) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_WQ.json" --outdir "$SCRIPT_DIR/outdir/WQ"
    echo -e "${GREEN}✓ WQ results saved${NC}"
    echo ""

    # EGAS on MGT dataset
    echo -e "${YELLOW}[6/8]${NC} Running EGAS search on MAGIC Gamma Telescope (MGT) dataset..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/egas_MGT.json" --outdir "$SCRIPT_DIR/outdir/MGT"
    echo -e "${GREEN}✓ MGT results saved${NC}"
    echo ""
fi

if [ "$RUN_PHOTONIC" = "yes" ]; then
    # Photonic implementation
    echo -e "${YELLOW}[7/8]${NC} Running photonic QKSVM on MAGIC Gamma Telescope (MGT)..."
    cd "$REPO_ROOT"
    $PYTHON implementation.py --paper-dir "$SCRIPT_DIR" --config "$SCRIPT_DIR/configs/photonic_MGT.json" --outdir "$SCRIPT_DIR/outdir/photonic_MGT"
    echo -e "${GREEN}✓ Photonic results saved${NC}"
    echo ""
fi

# Generate plots
echo -e "${YELLOW}[8/8]${NC} Generating summary plots..."
cd "$REPO_ROOT"
if [ -f "$SCRIPT_DIR/utils/plot_results.py" ]; then
    # Create combined results directory
    mkdir -p "$SCRIPT_DIR/results"
    
    # Note: Uncomment and adjust plot generation commands when plot_results.py is available
    # $PYTHON "$SCRIPT_DIR/utils/plot_results.py" \
    #     --wasserstein "$SCRIPT_DIR/outdir/wasserstein/metrics.json" \
    #     --egas "$SCRIPT_DIR/outdir/PW/run_*/metrics.json" \
    #             "$SCRIPT_DIR/outdir/WQ/run_*/metrics.json" \
    #             "$SCRIPT_DIR/outdir/MGT/run_*/metrics.json" \
    #     --output "$SCRIPT_DIR/results/"
    
    echo -e "${YELLOW}Note: Plot generation requires plot_results.py configuration${NC}"
else
    echo -e "${YELLOW}Note: plot_results.py not found, skipping plot generation${NC}"
fi
echo -e "${GREEN}✓ Done${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  • Wasserstein:  $SCRIPT_DIR/outdir/wasserstein/"
echo "  • Fig 1:        $SCRIPT_DIR/outdir/fig1/"
if [ "$RUN_GATE_EGAS" = "yes" ]; then
    echo "  • EGAS (PW):    $SCRIPT_DIR/outdir/PW/"
    echo "  • EGAS (WQ):    $SCRIPT_DIR/outdir/WQ/"
    echo "  • EGAS (MGT):   $SCRIPT_DIR/outdir/MGT/"
fi
if [ "$RUN_PHOTONIC" = "yes" ]; then
    echo "  • Photonic:     $SCRIPT_DIR/outdir/photonic_MGT/"
fi
echo ""
echo "Usage:"
echo "  ./run_all_experiments.sh               # Run all (default)"
echo "  ./run_all_experiments.sh no yes no    # Skip tests, run gate EGAS only"
echo "  ./run_all_experiments.sh yes no yes   # Skip gate EGAS, run tests & photonic"
