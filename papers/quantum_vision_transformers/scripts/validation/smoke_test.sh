#!/usr/bin/env bash
# smoke_test.sh — 3-epoch, 1-layer sanity test of every model.
# Usage:  bash scripts/validation/smoke_test.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS=(A B C D E F)
PASSED=0
FAILED=0

for m in "${MODELS[@]}"; do
    cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
    if [ ! -f "$cfg" ]; then
        # D_full doesn't have a lowercase-letter config
        if [ "$m" = "D_full" ]; then cfg="configs/model_d_full_retina.json"; fi
    fi
    [ -f "$cfg" ] || { echo "SKIP $m ($cfg not found)"; continue; }

    out="outdir/smoke_${m}"
    if [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ]; then
        echo "SKIP smoke_${m} (folder exists; set FORCE_RERUN=1 to rerun)"
        continue
    fi
    echo ""
    echo "════ Smoke: Model ${m} (3 ep, 1 layer) ════"

    tmp=$(mktemp)
    python -c "
import json
with open('${cfg}') as f: c = json.load(f)
c['epochs'] = 3; c['n_layers'] = 1
json.dump(c, open('${tmp}', 'w'), indent=2)
"
    if python implementation.py --config "$tmp" --seed 42 --outdir "$out" 2>&1 | tail -8; then
        if [ -f "${out}/results.json" ]; then
            echo "  ✓ Model ${m} OK"
            PASSED=$((PASSED + 1))
        else
            echo "  ✗ Model ${m} — no results.json"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ✗ Model ${m} — crashed"
        FAILED=$((FAILED + 1))
    fi
    rm -f "$tmp"
done

echo ""
echo "════════════════════════════════════════"
echo "  Smoke test: ${PASSED} passed, ${FAILED} failed"
echo "════════════════════════════════════════"
