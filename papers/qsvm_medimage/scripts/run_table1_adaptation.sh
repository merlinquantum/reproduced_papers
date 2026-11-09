#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DATA_PATH="${DATA_PATH:-../../data/qsvm_medimage/pneumoniamnist_train.pkl}"
RESULT_ROOT="${RESULT_ROOT:-outdir/table1_adaptation}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
Q_VALUES_CSV="${Q_VALUES:-2,4,6}"
SEEDS_CSV="${SEEDS:-0,1,2,3,4}"
C_VALUES="${C_VALUES:-0.01,0.1,1,10,100}"
CIRCUIT_SEED="${CIRCUIT_SEED:-0}"
LEAKAGE_MODES_CSV="${LEAKAGE_MODES:-legacy,train_only}"
TRACE_MODES_CSV="${TRACE_MODES:-legacy_square_only,train_trace}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qsvm-medimage-matplotlib}"
XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/qsvm-merlin-data}"

export MPLCONFIGDIR
export XDG_DATA_HOME
export PYTHONDONTWRITEBYTECODE=1

IFS=',' read -r -a Q_VALUES_ARRAY <<< "$Q_VALUES_CSV"
IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS_CSV"
IFS=',' read -r -a LEAKAGE_MODES_ARRAY <<< "$LEAKAGE_MODES_CSV"
IFS=',' read -r -a TRACE_MODES_ARRAY <<< "$TRACE_MODES_CSV"

for leakage_mode in "${LEAKAGE_MODES_ARRAY[@]}"; do
  leakage_args=(--data_path "$DATA_PATH")
  case "$leakage_mode" in
    legacy)
      leakage_id="legacy_leak"
      ;;
    train_only)
      leakage_id="train_only"
      leakage_args+=(--fix_leakage)
      ;;
    *)
      echo "Unsupported leakage mode: $leakage_mode" >&2
      exit 2
      ;;
  esac

  echo "Classical baselines: leakage=$leakage_mode"
  "$PYTHON_BIN" scripts/classical_svm_c1_pca.py \
    "${leakage_args[@]}" \
    --output_dir "$RESULT_ROOT/classical/$leakage_mode/linear_c1" \
    --pca_dims "$Q_VALUES_CSV" \
    --kernels linear \
    --c_values 1.0 \
    --seeds "$SEEDS_CSV" \
    --max_samples "$MAX_SAMPLES"

  "$PYTHON_BIN" scripts/classical_svm_c1_pca.py \
    "${leakage_args[@]}" \
    --output_dir "$RESULT_ROOT/classical/$leakage_mode/rbf_tuned" \
    --pca_dims "$Q_VALUES_CSV" \
    --kernels rbf \
    --c_values "$C_VALUES" \
    --seeds "$SEEDS_CSV" \
    --max_samples "$MAX_SAMPLES"
done

for trace_mode in "${TRACE_MODES_ARRAY[@]}"; do
  case "$trace_mode" in
    legacy_square_only)
      trace_id="legacy_trace"
      merlin_normalization="none"
      ;;
    train_trace)
      trace_id="train_trace"
      merlin_normalization="train_trace"
      ;;
    *)
      echo "Unsupported trace mode: $trace_mode" >&2
      exit 2
      ;;
  esac

  for leakage_mode in "${LEAKAGE_MODES_ARRAY[@]}"; do
    leakage_args=(--data_path "$DATA_PATH")
    case "$leakage_mode" in
      legacy)
        leakage_id="legacy_leak"
        ;;
      train_only)
        leakage_id="train_only"
        leakage_args+=(--fix_leakage)
        ;;
    esac

    protocol_id="${leakage_id}__${trace_id}"
    protocol_root="$RESULT_ROOT/protocols/$protocol_id"

    for q_value in "${Q_VALUES_ARRAY[@]}"; do
      for seed_value in "${SEEDS_ARRAY[@]}"; do
        echo "QSVM: protocol=$protocol_id q=$q_value seed=$seed_value"
        "$PYTHON_BIN" scripts/qsvm_cuda_embeddings_insurance.py \
          "${leakage_args[@]}" \
          --output_dir "$protocol_root/qsvm/q_${q_value}/seed_${seed_value}" \
          --backend cpu \
          --qubits "$q_value" \
          --max_samples "$MAX_SAMPLES" \
          --single_mode \
          --num_seeds 1 \
          --seed "$seed_value" \
          --trace_protocol "$trace_mode"

        echo "MerLin: protocol=$protocol_id PCA=$q_value seed=$seed_value"
        "$PYTHON_BIN" scripts/merlin_fidelity_kernel.py \
          "${leakage_args[@]}" \
          --output_dir "$protocol_root/merlin/q_${q_value}/seed_${seed_value}" \
          --pca_dim "$q_value" \
          --seed "$seed_value" \
          --circuit_seed "$CIRCUIT_SEED" \
          --max_samples "$MAX_SAMPLES" \
          --kernel_normalization "$merlin_normalization"
      done
    done
  done
done

"$PYTHON_BIN" scripts/aggregate_protocol_matrix.py \
  --result_root "$RESULT_ROOT"

legacy_protocol_root="$RESULT_ROOT/protocols/legacy_leak__legacy_trace"
if [[ -d "$legacy_protocol_root/qsvm" ]]; then
  "$PYTHON_BIN" scripts/create_table1_adaptation.py \
    --qsvm_dirs "$legacy_protocol_root/qsvm" \
    --tier1_csv "$RESULT_ROOT/classical/legacy/linear_c1/metrics_summary.csv" \
    --tier2_csv "$RESULT_ROOT/classical/legacy/rbf_tuned/metrics_summary.csv" \
    --output "$RESULT_ROOT/table1_cpu_pneumoniamnist.png"
fi

echo "Protocol matrix: $RESULT_ROOT/protocol_summary.md"
