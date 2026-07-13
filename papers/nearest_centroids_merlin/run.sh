#!/bin/bash
# Reproduce all experiments from Johri et al. (2020)
# "Nearest Centroid Classification on a Trapped Ion Quantum Computer"
#
# Usage: From repo root, run:
#   bash papers/nearest_centroids_merlin/run.sh
#
# Run outputs go to outdir/ (gitignored). Final reproduced artifacts
# (figures, tables, summary results) are copied to results/ (tracked).

set -e

PROJECT="papers/nearest_centroids_merlin"
CONFIGS="$PROJECT/configs"
OUTDIR="$PROJECT/outdir"
RESULTS="$PROJECT/results"

echo "=========================================="
echo "Reproducing Paper Figures"
echo "=========================================="

# Figure 8: Synthetic Data
echo ""
echo "--- Figure 8: Synthetic Data ---"
python implementation.py --project $PROJECT --config $CONFIGS/synthetic_4q_2c.json
python implementation.py --project $PROJECT --config $CONFIGS/synthetic_4q_4c.json
python implementation.py --project $PROJECT --config $CONFIGS/synthetic_8q_2c.json
python implementation.py --project $PROJECT --config $CONFIGS/synthetic_8q_4c.json

# Figure 9: IRIS Dataset
echo ""
echo "--- Figure 9: IRIS Dataset ---"
python implementation.py --project $PROJECT --config $CONFIGS/iris_ns100.json
python implementation.py --project $PROJECT --config $CONFIGS/iris_ns500.json
python implementation.py --project $PROJECT --config $CONFIGS/iris_ns1000.json

# Figure 11: MNIST Dataset
echo ""
echo "--- Figure 11: MNIST Dataset ---"
python implementation.py --project $PROJECT --config $CONFIGS/mnist_0v1.json
python implementation.py --project $PROJECT --config $CONFIGS/mnist_2v7.json
python implementation.py --project $PROJECT --config $CONFIGS/mnist_4class.json
python implementation.py --project $PROJECT --config $CONFIGS/mnist_10class.json

echo ""
echo "=========================================="
echo "Collecting reproduced artifacts into results/"
echo "=========================================="

mkdir -p "$RESULTS/figures"

FOUND=0
for DONE_FILE in $(find "$OUTDIR" -name "done.txt" 2>/dev/null); do
    RUN_DIR=$(dirname "$DONE_FILE")
    FOUND=1
    echo "Collecting from: $RUN_DIR"

    # Copy figures
    if [ -d "$RUN_DIR/figures" ]; then
        cp -r "$RUN_DIR/figures/"* "$RESULTS/figures/" 2>/dev/null && \
            echo "  -> figures/"
    fi

    # Copy per-experiment result JSONs
    for f in "$RUN_DIR"/results_*.json; do
        [ -f "$f" ] && cp "$f" "$RESULTS/" && echo "  -> $(basename "$f")"
    done

    # Copy summary results (named by run to avoid clobbering)
    if [ -f "$RUN_DIR/summary_results.json" ]; then
        RUN_NAME=$(basename "$RUN_DIR")
        cp "$RUN_DIR/summary_results.json" "$RESULTS/summary_${RUN_NAME}.json"
        echo "  -> summary_${RUN_NAME}.json"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "Warning: no completed runs found in $OUTDIR"
    echo "Run outputs are in $OUTDIR — copy artifacts to $RESULTS manually."
else
    echo ""
    echo "Reproduced artifacts are in: $RESULTS"
fi

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="