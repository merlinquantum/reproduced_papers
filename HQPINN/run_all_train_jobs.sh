#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

shopt -s nullglob

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash HQPINN/run_all_train_jobs.sh
  bash HQPINN/run_all_train_jobs.sh --dry-run

Runs HQPINN training configs sequentially, starting with classical-classical jobs.
Includes `dho-pp` and `dho-cp`, but skips the other `-cp` and SEE/DEE/TAF `-pp` jobs.
Order is grouped to start with classical-classical models.
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

# 1) DHO jobs first.
add_path "HQPINN/configs/dho_cc_train.json"
add_path "HQPINN/configs/dho_cperc_train.json"
add_path "HQPINN/configs/dho_ci_train.json"
add_path "HQPINN/configs/dho_cp_train.json"
add_path "HQPINN/configs/dho_ii_train.json"
add_path "HQPINN/configs/dho_percperc_train.json"
add_path "HQPINN/configs/dho_pp_train.json"

# 2) SEE jobs.
add_group "HQPINN/configs/see_cc_train_*.json" "HQPINN/configs/see_cc_train.json"
add_group "HQPINN/configs/see_ci_train_*.json" "HQPINN/configs/see_ci_train.json"
add_group "HQPINN/configs/see_ii_train_*.json" "HQPINN/configs/see_ii_train.json"

# 3) DEE jobs.
add_group "HQPINN/configs/dee_cc_train_*.json" "HQPINN/configs/dee_cc_train.json"
add_group "HQPINN/configs/dee_ci_train_*.json" "HQPINN/configs/dee_ci_train.json"
add_group "HQPINN/configs/dee_ii_train_*.json" "HQPINN/configs/dee_ii_train.json"

# 4) TAF jobs.
add_group "HQPINN/configs/taf_cc_train_*.json" "HQPINN/configs/taf_cc_train.json"
add_group "HQPINN/configs/taf_ci_train_*.json" "HQPINN/configs/taf_ci_train.json"
add_group "HQPINN/configs/taf_ii_train_*.json" "HQPINN/configs/taf_ii_train.json"

total_jobs="${#jobs[@]}"
if (( total_jobs == 0 )); then
    echo "No training jobs found." >&2
    exit 1
fi

printf 'Queued %d HQPINN training jobs.\n' "${total_jobs}"

for idx in "${!jobs[@]}"; do
    job="${jobs[$idx]}"
    printf '\n[%d/%d] %s\n' "$((idx + 1))" "${total_jobs}" "${job}"
    cmd=("${PYTHON_BIN}" -m HQPINN --config "${job}")

    if (( DRY_RUN == 1 )); then
        printf 'DRY RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    "${cmd[@]}"
done

printf '\nCompleted %d HQPINN training jobs.\n' "${total_jobs}"
