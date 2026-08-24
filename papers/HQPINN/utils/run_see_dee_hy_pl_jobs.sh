#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${PAPER_ROOT}/../.." && pwd)"
cd "${PAPER_ROOT}"

shopt -s nullglob

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash utils/run_see_dee_hy_pl_jobs.sh
  bash utils/run_see_dee_hy_pl_jobs.sh --dry-run

Runs the SEE and DEE hybrid PennyLane training configs sequentially.
The queue starts with SEE hy-pl configs, then DEE hy-pl configs.
EOF
}

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 1
fi

jobs=()

add_path() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing config: ${path}" >&2
        exit 1
    fi
    jobs+=("${path}")
}

add_group() {
    local preferred_pattern="$1"
    local fallback_path="${2:-}"
    local matches=(${preferred_pattern})

    if (( ${#matches[@]} > 0 )); then
        jobs+=("${matches[@]}")
        return
    fi

    if [[ -n "${fallback_path}" ]]; then
        add_path "${fallback_path}"
        return
    fi

    echo "No configs matched: ${preferred_pattern}" >&2
    exit 1
}

add_group "configs/see_hy_pl_train_*.json" "configs/see_hy_pl_train.json"
add_group "configs/dee_hy_pl_train_*.json" "configs/dee_hy_pl_train.json"

total_jobs="${#jobs[@]}"
if (( total_jobs == 0 )); then
    echo "No SEE/DEE hy-pl training jobs found." >&2
    exit 1
fi

printf 'Queued %d SEE/DEE hy-pl training jobs.\n' "${total_jobs}"

for idx in "${!jobs[@]}"; do
    job="${jobs[$idx]}"
    printf '\n[%d/%d] %s\n' "$((idx + 1))" "${total_jobs}" "${job}"
    cmd=("${PYTHON_BIN}" "${REPO_ROOT}/implementation.py" --paper HQPINN --config "${job}")

    if (( DRY_RUN == 1 )); then
        printf 'DRY RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    "${cmd[@]}"
done

printf '\nCompleted %d SEE/DEE hy-pl training jobs.\n' "${total_jobs}"
