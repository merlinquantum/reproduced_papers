#!/usr/bin/env bash
# run_all_ansatz_2_to_9b.sh
# Runs Table-1 columns for ansatz: 2 3 4 5 6 7 8 9a 9b
# Uses: run_experiment.py (in the current directory)

set -u -o pipefail  # keep going on individual failures; report them

PY=python
SCRIPT="run_experiment.py"

DATASET="mnist"
CLASSES="0,1"
SEEDS=1
COST="cross_entropy"
CIRCUIT="QCNN"

# If you hit Keras/TF issues on autoencoder encodings, this helps:
export KERAS_BACKEND=tensorflow

ANSATZ_LIST=(7 8 9a 9b)

# Encodings for each Table-1 column:
# - AE:            resize256
# - Qubit (PCA8):  pca8
# - Qubit (AE8):   autoencoder8
# - Dense (PCA16): pca16-1         # if your repo has *-compact, you can switch to pca16-compact
# - Dense (AE16):  autoencoder16-1  # likewise, switch to autoencoder16-compact if available
# - HDE (PCA32):   pca32-1
# - HDE (AE32):    autoencoder32-1
# - HAE (PCA30):   pca30-1
# - HAE (AE30):    autoencoder30-1
ENCODINGS=(resize256 pca8 autoencoder8 pca16-1 autoencoder16-1 pca32-1 autoencoder32-1 pca30-1 autoencoder30-1)

run_one() {
  local ansatz="$1"
  local encoding="$2"
  echo "==> Ansatz ${ansatz} | Encoding ${encoding}"
  set +e
  $PY "$SCRIPT" \
    --dataset "$DATASET" \
    --classes "$CLASSES" \
    --ansatz "$ansatz" \
    --encoding "$encoding" \
    --seeds "$SEEDS" \
    --cost_fn "$COST" \
    --circuit "$CIRCUIT"
  ec=$?
  set -e
  if [[ $ec -ne 0 ]]; then
    echo "   [FAILED] ansatz=$ansatz encoding=$encoding (exit $ec)"
  else
    echo "   [OK]     ansatz=$ansatz encoding=$encoding"
  fi
  echo
}

main() {
  for ans in "${ANSATZ_LIST[@]}"; do
    for enc in "${ENCODINGS[@]}"; do
      run_one "$ans" "$enc"
    done
  done
  echo "All runs attempted."
}

main
