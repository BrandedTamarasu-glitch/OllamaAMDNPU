#!/usr/bin/env bash
# bench-power.sh — measure average SoC power (PPT) during llama-bench inference
# Usage: bench-power.sh --npu|--vulkan
#
# NPU mode:    hides libggml-vulkan.so; uses GGML_BACKEND_PATH for XDNA.
# Vulkan mode: hides libggml-xdna.so; sets GGML_BACKEND_PATH to Vulkan .so.
#
# Uses llama-bench (stdout output) to avoid llama-cli tty-write issue.
# Reads amdgpu PPT (hwmon3/power1_input, µW) at 200ms intervals and
# reports: avg watts, t/s, and energy per decode token.

set -euo pipefail

POWER_FILE=/sys/class/hwmon/hwmon3/power1_input
MODEL=${MODEL:-models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}
PP_TOKENS=${PP_TOKENS:-160}   # simulated prompt tokens for llama-bench
TG_TOKENS=${TG_TOKENS:-30}    # decode tokens for llama-bench
VULKAN_SO=./build/bin/libggml-vulkan.so
VULKAN_SO_HIDDEN=./build/bin/libggml-vulkan.so.bench-hidden
XDNA_SO=./build/bin/libggml-xdna.so
XDNA_SO_HIDDEN=./build/bin/libggml-xdna.so.bench-hidden

if [[ ! -f "$POWER_FILE" ]]; then
    echo "ERROR: $POWER_FILE not found" >&2; exit 1
fi

MODE="${1:-}"
if [[ "$MODE" != "--npu" && "$MODE" != "--vulkan" ]]; then
    echo "Usage: bench-power.sh --npu|--vulkan" >&2; exit 1
fi

echo "=== bench-power: ${MODE/--/} mode ==="
echo "Model: $MODEL  (pp=${PP_TOKENS} tg=${TG_TOKENS})"
echo ""

POWER_LOG=$(mktemp /tmp/bench-power-XXXXXX.log)
VULKAN_HIDDEN=0

XDNA_HIDDEN=0

cleanup() {
    [[ "$VULKAN_HIDDEN" -eq 1 && -f "$VULKAN_SO_HIDDEN" ]] && mv "$VULKAN_SO_HIDDEN" "$VULKAN_SO"
    [[ "$XDNA_HIDDEN"   -eq 1 && -f "$XDNA_SO_HIDDEN"   ]] && mv "$XDNA_SO_HIDDEN"   "$XDNA_SO"
    rm -f "$POWER_LOG"
}
trap cleanup EXIT

# Isolate backends: NPU hides Vulkan; Vulkan hides XDNA
if [[ "$MODE" == "--npu" ]]; then
    if [[ -f "$VULKAN_SO" ]]; then mv "$VULKAN_SO" "$VULKAN_SO_HIDDEN"; VULKAN_HIDDEN=1; fi
elif [[ "$MODE" == "--vulkan" ]]; then
    if [[ -f "$XDNA_SO" ]]; then mv "$XDNA_SO" "$XDNA_SO_HIDDEN"; XDNA_HIDDEN=1; fi
fi

# Start power sampler
( while true; do cat "$POWER_FILE"; sleep 0.2; done ) > "$POWER_LOG" &
SAMPLER_PID=$!

# Run llama-bench — writes t/s to stdout (avoids llama-cli tty-write issue).
# -r 1: single repetition (no multi-run averaging needed; power log handles stats).
if [[ "$MODE" == "--vulkan" ]]; then
    # Explicitly set GGML_BACKEND_PATH to Vulkan .so; -ngl 99 for full GPU offload.
    INFER_OUT=$(
        GGML_BACKEND_PATH=$(realpath ./build/bin/libggml-vulkan.so) \
            ./build/bin/llama-bench \
            -m "$MODEL" -p "$PP_TOKENS" -n "$TG_TOKENS" -r 1 -ngl 99 2>/dev/null
    )
else
    # NPU: GGML_BACKEND_PATH already set from caller's environment (XDNA .so).
    INFER_OUT=$(
        ./build/bin/llama-bench \
            -m "$MODEL" -p "$PP_TOKENS" -n "$TG_TOKENS" -r 1 2>/dev/null
    )
fi

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

# Parse t/s from llama-bench markdown table:
#   | model | size | params | backend | ngl | test    | t/s          |
#   | ...   | ...  | ...    | ...     | ... | pp 160  | 315.89 ± ... |
#   | ...   | ...  | ...    | ...     | ... | tg 30   |  49.22 ± ... |
# NF-1 is the t/s column (NF is empty field after trailing |).
PROMPT_TS=$(echo "$INFER_OUT" | awk -F'|' \
    '/pp[0-9]/ { t=$(NF-1); gsub(/^ +/,"",t); print t+0 }' || true)
GEN_TS=$(echo "$INFER_OUT" | awk -F'|' \
    '/tg[0-9]/ { t=$(NF-1); gsub(/^ +/,"",t); print t+0 }' || true)
[[ -z "$PROMPT_TS" ]] && PROMPT_TS="n/a"
[[ -z "$GEN_TS" ]]    && GEN_TS="n/a"

# Power stats via awk (µW → W)
N_SAMPLES=$(wc -l < "$POWER_LOG")
if [[ "$N_SAMPLES" -gt 0 ]]; then
    read AVG_W MIN_W MAX_W < <(awk '
        BEGIN { sum=0; min=999999999; max=0 }
        { sum+=$1; if($1<min) min=$1; if($1>max) max=$1 }
        END { printf "%.2f %.2f %.2f\n", sum/NR/1e6, min/1e6, max/1e6 }
    ' "$POWER_LOG")
else
    AVG_W="n/a"; MIN_W="n/a"; MAX_W="n/a"
fi

# Energy per decode token
if [[ "$GEN_TS" != "n/a" && "$AVG_W" != "n/a" ]]; then
    J_PER_TOK=$(awk "BEGIN { printf \"%.4f\", $AVG_W / $GEN_TS }")
else
    J_PER_TOK="n/a"
fi

echo "Results ($N_SAMPLES power samples):"
printf "  Prefill:         %s t/s\n" "$PROMPT_TS"
printf "  Decode:          %s t/s\n" "$GEN_TS"
printf "  Avg power (PPT): %s W  (min %s / max %s)\n" "$AVG_W" "$MIN_W" "$MAX_W"
printf "  Energy/token:    %s J/tok\n" "$J_PER_TOK"
echo ""
