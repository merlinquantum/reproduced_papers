#!/bin/bash
# Reproduce all experiments from Johri et al. (2020)
# "Nearest Centroid Classification on a Trapped Ion Quantum Computer"
#
# Usage: From repo root, run:
#   chmod +x papers/nearest_centroids_merlin/run.sh
#   ./papers/nearest_centroids_merlin/run.sh

set -e

PROJECT="papers/nearest_centroids_merlin"
CONFIGS="$PROJECT/configs"

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
echo "All experiments complete!"
echo "=========================================="