#!/usr/bin/env bash
# Smoke-test all reproduced papers by installing per-project deps, running the default config,
# and executing pytest. Suitable for macOS default bash (no mapfile/readarray dependency).
# Run from repo root: scripts/smoke_test_all_papers.sh

set -euo pipefail

# Ensure Python writes stdout/stderr without buffering so logs stream immediately
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.smoke_logs"
ENV_ROOT="$ROOT_DIR/.smoke_envs"
PYTHON_BIN=${PYTHON_BIN:-python3}

# Portable timeout helper: prefers GNU timeout if present, falls back to perl alarm.
timeout_cmd() {
  local seconds=$1
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$seconds" "$@"
  elif command -v perl >/dev/null 2>&1; then
    perl -e 'alarm shift @ARGV; exec @ARGV' "$seconds" "$@"
  else
    echo "[WARN] No timeout available; running without limit" >&2
    "$@"
  fi
}

mkdir -p "$LOG_DIR" "$ENV_ROOT"

cd "$ROOT_DIR"

# Gather papers via the shared runner (captures nested subprojects like fock_state_expressivity/*)
PAPERS=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  PAPERS+=("$line")
done < <($PYTHON_BIN implementation.py --list-papers)

# Optional substring filter to target specific papers quickly (first arg)
FILTER=${1:-}
filtered=()
if [[ -n "$FILTER" ]]; then
  for p in "${PAPERS[@]}"; do
    if [[ "$p" == *"$FILTER"* ]]; then
      filtered+=("$p")
    fi
  done
  PAPERS=("${filtered[@]-}")
fi

printf "Found %d papers\n" "${#PAPERS[@]}"

status=0
RESULT_PAPERS=()
RESULT_STATUS=()

for paper in "${PAPERS[@]}"; do
  paper_rel="$paper"
  # Normalize paths that already include the leading "papers/" prefix
  if [[ "$paper_rel" == papers/* ]]; then
    paper_rel="${paper_rel#papers/}"
  fi

  paper_sanitized=${paper_rel//\//_}
  env_dir="$ENV_ROOT/$paper_sanitized"
  log_file="$LOG_DIR/$paper_sanitized.log"

  paper_python="$PYTHON_BIN"
  if [[ "$paper_rel" == "qLLM" ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
      paper_python="python3.12"
    else
      echo "[WARN] python3.12 not found; qLLM installs may fail under $PYTHON_BIN." >&2
    fi
  fi

  echo "==> [$paper] setting up venv at $env_dir"
  keep_existing_venv=1
  
  if [[ $keep_existing_venv -eq 0 ]] || [[ ! -f "$env_dir/bin/activate" ]]; then
    if [[ -d "$env_dir" ]] || [[ -f "$env_dir/bin/activate" ]]; then
      echo "Old venv detected, then removed and new venv installing..."
      rm -rf "$env_dir"
    fi
    $paper_python -m venv "$env_dir"
  fi

  # shellcheck disable=SC1090
  source "$env_dir/bin/activate"
  pip install -U pip wheel >/dev/null

  # Install project-specific requirements if present
  req_file="$ROOT_DIR/papers/$paper_rel/requirements.txt"
  if [[ -f "$req_file" ]]; then
    echo "   installing $req_file"
    pip install -r "$req_file" >"$log_file" 2>&1 || true
  fi

  # Expose repo code without editable install to avoid setuptools flat-layout errors
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

  run_status="RUN_OK"
  echo "   running default experiment (logs: $log_file)"
  if timeout_cmd 120 "$env_dir/bin/python" "$ROOT_DIR/implementation.py" --paper "$paper_rel" >>"$log_file" 2>&1; then
    echo "   [$paper] run: SUCCESS"
  else
    rc=$?
    if [[ $rc -eq 124 || $rc -eq 142 ]]; then
      run_status="RUN_TIMEOUT"
      echo "[TIMEOUT] run exceeded 120s" >>"$log_file"
      echo "   [$paper] run: TIMEOUT (see $log_file)"
    else
      run_status="RUN_FAIL"
      echo "   [$paper] run: FAIL (see $log_file)"
    fi
    status=1
  fi

  test_status="TEST_OK"
  echo "   running pytest"
  if (cd "$ROOT_DIR/papers/$paper_rel" && timeout_cmd 120 "$env_dir/bin/python" -m pytest -q >>"$log_file" 2>&1); then
    echo "   [$paper] tests: SUCCESS"
  else
    rc=$?
    if [[ $rc -eq 124 || $rc -eq 142 ]]; then
      test_status="TEST_TIMEOUT"
      echo "[TIMEOUT] tests exceeded 120s" >>"$log_file"
      echo "   [$paper] tests: TIMEOUT (see $log_file)"
    else
      test_status="TEST_FAIL"
      echo "   [$paper] tests: FAIL (see $log_file)"
    fi
    status=1
  fi

  RESULT_PAPERS+=("$paper_rel")
  RESULT_STATUS+=("${run_status}|${test_status}")

done

printf "\nSummary:\n"
for i in "${!RESULT_PAPERS[@]}"; do
  printf " - %s: %s\n" "${RESULT_PAPERS[$i]}" "${RESULT_STATUS[$i]}"
done

exit $status
