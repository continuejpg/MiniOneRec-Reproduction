#!/bin/bash
# =============================================================================
# scripts/benchmark_inference.sh -- inference quality-latency benchmark
#
# Measures the quality-vs-cost trade-off of the generative recommender across
# beam width and batch size, on the FORMAL clean SFT model.
#
#   quality sweep    : beam in {5,10,20,50}, batch 8, full test (4533)
#   efficiency sweep : beam 20, batch in {1,8,32}, first 512 test rows
#
# Protocol identity: the benchmark calls evaluate.py's OWN main(), so prompts,
# SID parsing, the constrained-decoding trie (ORIGINAL INFO) and the logits
# processor are exactly the formal ones. Nothing is re-implemented.
#
# Output: $RUN_ROOT/inference_benchmark/{results.json,results.csv,...}
#         Refuses to run if that directory already exists (never overwrites).
#
# Does NOT train, does NOT touch the formal eval outputs, does NOT modify
# analysis/results/.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# main benchmark model: the formal clean SFT checkpoint
MAIN_MODEL="$RUN_ROOT/industrial_sft/final_checkpoint"
OUT="$RUN_ROOT/inference_benchmark"

MODE="${MODE:-all}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

echo "=============================================================================="
echo " Inference quality-latency benchmark"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   RUN_ROOT     = $RUN_ROOT"
echo "   CATEGORY     = $CATEGORY"
echo "   MAIN_MODEL   = $MAIN_MODEL"
echo "   OUT          = $OUT"
echo "   MODE         = $MODE"
echo "=============================================================================="
echo

# -----------------------------------------------------------------------------
# read-only preflight (no GPU work)
# -----------------------------------------------------------------------------
fail=0
chk() { if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi; }
refuse() { printf '  REFUSE  %s\n' "$1"; fail=1; }

echo "=== preflight ==="
chk "$PROJECT_ROOT/evaluate.py"
chk "$PROJECT_ROOT/calc.py"
chk "$PROJECT_ROOT/analysis/benchmark_inference.py"
chk "$INFO"
chk "$TEST"
chk "$ITEM_META"

# the main benchmark model MUST be the formal clean SFT run
if [ -d "$MAIN_MODEL" ]; then
    chk "$MAIN_MODEL/config.json"
    chk "$MAIN_MODEL/model.safetensors"
else
    echo "  MISSING $MAIN_MODEL"
    echo "          the benchmark targets the FORMAL clean SFT checkpoint;"
    echo "          it must be restored before this can run. Refusing to"
    echo "          substitute GRPO or shuffled models."
    fail=1
fi

# output must not exist: never silently overwrite a previous benchmark
if [ -e "$OUT" ]; then
    refuse "output dir already exists: $OUT (move it aside before rerunning)"
else
    echo "  OK      output dir is free: $OUT"
fi

# benchmark must not touch the committed provenance
if grep -q 'analysis/results' "$PROJECT_ROOT/analysis/benchmark_inference.py"; then
    refuse "analysis/benchmark_inference.py references analysis/results/"
else
    echo "  OK      benchmark does not reference analysis/results/"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    refuse "nvidia-smi not found -- no GPU on this host"
fi

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting benchmark ==="
    if [ "$PREFLIGHT_ONLY" = "1" ]; then
        echo
        echo "--- static checks only (no GPU required) ---"
        "$PY" "$PROJECT_ROOT/analysis/benchmark_inference.py" --preflight
        exit $?
    fi
    exit 1
fi
echo "=== preflight OK ==="
echo

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "PREFLIGHT_ONLY=1 -- running static checks and stopping."
    "$PY" "$PROJECT_ROOT/analysis/benchmark_inference.py" --preflight
    exit $?
fi

# -----------------------------------------------------------------------------
# benchmark
# -----------------------------------------------------------------------------
"$PY" -u "$PROJECT_ROOT/analysis/benchmark_inference.py" --mode "$MODE" --out-dir "$OUT"
rc=$?

echo
echo "=== done (exit $rc) ==="
if [ -f "$OUT/results.csv" ]; then
    echo "--- results.csv ---"
    cat "$OUT/results.csv"
fi
exit $rc
