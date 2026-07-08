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

PROJECT="papers/bandwidth_tunning"
CONFIGS="$PROJECT/configs"
OUTDIR="$PROJECT/outdir"
RESULTS="$PROJECT/results"

echo "=========================================="
echo "Reproducing Paper Figures"
echo "=========================================="

# Figure 3: Plasticc
echo ""
echo "--- Figure 3: Plasticc dataset ---"
python implementation.py --project $PROJECT --config $CONFIGS/fig3.1-plasticc.json
python implementation.py --project $PROJECT --config $CONFIGS/fig3.2-plasticc.json

# Figure 3: KMNIST28
echo ""
echo "--- Figure 3: kmnist28 dataset---"
python implementation.py --project $PROJECT --config $CONFIGS/fig3.1-kmnist28.json
python implementation.py --project $PROJECT --config $CONFIGS/fig3.2-kmnist28.json

# Figure 3: FashionMNIST
echo ""
echo "--- Figure 3: FashionMNIST Dataset ---"
python implementation.py --project $PROJECT --config $CONFIGS/fig3.1-fashionmnist.json
python implementation.py --project $PROJECT --config $CONFIGS/fig3.2-fashionmnist.json

# Figure 3: Hidden Manifold
echo ""
echo "--- Figure 3: Hidden Manifold Dataset ---"
python implementation.py --project $PROJECT --config $CONFIGS/fig3.1-hidden_manifold.json
python implementation.py --project $PROJECT --config $CONFIGS/fig3.2-hidden_manifold.json


