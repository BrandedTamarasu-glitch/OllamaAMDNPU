#!/usr/bin/env bash
# bench-power.sh — measure average SoC power (PPT) during llama-bench inference
# Usage: bench-power.sh --npu|--vulkan|--phase7 [--json]
#
# NPU mode:     hides libggml-vulkan.so; uses GGML_BACKEND_PATH for XDNA.
#               Measures NPU-only prefill+decode (p=PP_TOKENS, n=TG_TOKENS).
# Vulkan mode:  hides libggml-xdna.so; sets GGML_BACKEND_PATH to Vulkan .so.
#               Measures Vulkan-only prefill+decode.
# Phase7 mode:  both backends active; decode-only run (p=0, n=TG_TOKENS, ub=1).
#               Measures NPU decode power with Vulkan idle.
#               Set GGML_XDNA_MIN_N=1 GGML_XDNA_MAX_N=1 before running.
#
# Uses llama-bench (stdout output) to avoid llama-cli tty-write issue.
# Reads amdgpu PPT (hwmon3/power1_input, µW) at 50ms intervals and
# reports: avg watts, t/s, and energy per decode token.
#
# Flags:
#   --json   Emit structured JSON output in addition to human-readable summary.

set -euo pipefail

# Auto-detect the amdgpu hwmon directory (index can shift across boots).
POWER_FILE=""
for hwmon_dir in /sys/class/hwmon/hwmon*; do
    if [[ -f "$hwmon_dir/name" ]] && grep -q amdgpu "$hwmon_dir/name" 2>/dev/null; then
        if [[ -f "$hwmon_dir/power1_input" ]]; then
            POWER_FILE="$hwmon_dir/power1_input"
            break
        fi
    fi
done
if [[ -z "$POWER_FILE" ]]; then
    POWER_FILE=/sys/class/hwmon/hwmon3/power1_input  # fallback
fi
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
if [[ "$MODE" != "--npu" && "$MODE" != "--vulkan" && "$MODE" != "--phase7" ]]; then
    echo "Usage: bench-power.sh --npu|--vulkan|--phase7 [--json]" >&2; exit 1
fi

JSON_OUTPUT=0
shift  # consume mode arg
for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUTPUT=1 ;;
    esac
done

# When --json: route all human-readable output to stderr; JSON stays on stdout
[[ $JSON_OUTPUT -eq 1 ]] && exec 3>&1 1>&2

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

# Isolate backends: NPU hides Vulkan; Vulkan hides XDNA; phase7 runs both
if [[ "$MODE" == "--npu" ]]; then
    if [[ -f "$VULKAN_SO" ]]; then mv "$VULKAN_SO" "$VULKAN_SO_HIDDEN"; VULKAN_HIDDEN=1; fi
elif [[ "$MODE" == "--vulkan" ]]; then
    if [[ -f "$XDNA_SO" ]]; then mv "$XDNA_SO" "$XDNA_SO_HIDDEN"; XDNA_HIDDEN=1; fi
fi
# phase7: no hiding — both backends stay active

# Start initial power sampler (will be killed after warmup)
( while true; do cat "$POWER_FILE"; sleep 0.05; done ) > "$POWER_LOG" &
SAMPLER_PID=$!

# Warmup: run 5 seconds of inference to stabilize thermals
echo "Warming up (5s)..."
timeout 5 ./build/bin/llama-bench -m "$MODEL" -p 0 -n 5 -r 1 2>/dev/null || true

# Kill warmup sampler, discard warmup power data, restart fresh
kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null || true
> "$POWER_LOG"

# Idle baseline: measure 5 seconds of idle power (100 samples at 50ms = 5s)
echo "Measuring idle baseline (5s)..."
IDLE_LOG=$(mktemp)
for i in $(seq 100); do
    cat "$POWER_FILE" >> "$IDLE_LOG"
    sleep 0.05
done
IDLE_W=$(awk '{s+=$1; n++} END {printf "%.2f", s/n/1000000}' "$IDLE_LOG")
rm -f "$IDLE_LOG"
echo "Idle baseline: ${IDLE_W}W"

# Restart power sampler for actual measurement
( while true; do cat "$POWER_FILE"; sleep 0.05; done ) > "$POWER_LOG" &
SAMPLER_PID=$!

# Run llama-bench — writes t/s to stdout (avoids llama-cli tty-write issue).
# -r 1: single repetition (no multi-run averaging needed; power log handles stats).
START_MS=$(date +%s%3N)
if [[ "$MODE" == "--vulkan" ]]; then
    # Explicitly set GGML_BACKEND_PATH to Vulkan .so; -ngl 99 for full GPU offload.
    INFER_OUT=$(
        GGML_BACKEND_PATH=$(realpath ./build/bin/libggml-vulkan.so) \
            ./build/bin/llama-bench \
            -m "$MODEL" -p "$PP_TOKENS" -n "$TG_TOKENS" -r 1 -ngl 99 2>/dev/null
    )
elif [[ "$MODE" == "--phase7" ]]; then
    # Phase 7: both backends active; decode-only run at ub=1 to isolate NPU decode power.
    # Requires MIN_N=1 MAX_N=1 in environment (set in .zshrc for Phase 7).
    INFER_OUT=$(
        ./build/bin/llama-bench \
            -m "$MODEL" -p 0 -n "$TG_TOKENS" -ub 1 -r 1 2>/dev/null
    )
else
    # NPU: GGML_BACKEND_PATH already set from caller's environment (XDNA .so).
    INFER_OUT=$(
        ./build/bin/llama-bench \
            -m "$MODEL" -p "$PP_TOKENS" -n "$TG_TOKENS" -r 1 2>/dev/null
    )
fi
END_MS=$(date +%s%3N)
DIFF_MS=$(( END_MS - START_MS ))
DURATION=$(awk "BEGIN { printf \"%.1f\", $DIFF_MS / 1000 }")

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

# Energy per decode token (total and marginal)
ENERGY_JTOK="n/a"
MARGINAL_W="n/a"
MARGINAL_JTOK="n/a"
if [[ "$GEN_TS" != "n/a" && "$AVG_W" != "n/a" ]]; then
    ENERGY_JTOK=$(awk "BEGIN { printf \"%.4f\", $AVG_W / $GEN_TS }")
    MARGINAL_W=$(awk "BEGIN { printf \"%.2f\", $AVG_W - $IDLE_W }")
    MARGINAL_JTOK=$(awk "BEGIN { printf \"%.4f\", ($AVG_W - $IDLE_W) / $GEN_TS }")
fi

echo "Results ($N_SAMPLES power samples, ${DURATION}s):"
printf "  Prefill:         %s t/s\n" "$PROMPT_TS"
printf "  Decode:          %s t/s\n" "$GEN_TS"
printf "  Idle baseline:   %s W\n" "$IDLE_W"
printf "  Avg power (PPT): %s W  (min %s / max %s)\n" "$AVG_W" "$MIN_W" "$MAX_W"
printf "  Marginal power:  %s W\n" "$MARGINAL_W"
printf "  Energy/token:    %s J/tok (total)  %s J/tok (marginal)\n" "$ENERGY_JTOK" "$MARGINAL_JTOK"
echo ""

# Restore stdout for JSON block
[[ $JSON_OUTPUT -eq 1 ]] && exec 1>&3 3>&-

if [[ $JSON_OUTPUT -eq 1 ]]; then
    cat <<JSON_EOF
{
  "timestamp": "$(date -Iseconds)",
  "backend": "$MODE",
  "mode": "decode",
  "model": "$MODEL",
  "tile_config": "${GGML_XDNA_TILE_M:-?}x${GGML_XDNA_TILE_K:-?}x${GGML_XDNA_TILE_N:-?}",
  "xclbin_hash": "$(sha256sum "${GGML_XDNA_XCLBIN_PATH:-/dev/null}" 2>/dev/null | head -c 12 || echo "n/a")",
  "warmup_s": 5,
  "measurement_s": $DURATION,
  "sampling_ms": 50,
  "idle_power_w": $IDLE_W,
  "avg_power_w": $AVG_W,
  "marginal_power_w": $MARGINAL_W,
  "min_power_w": $MIN_W,
  "max_power_w": $MAX_W,
  "decode_ts": $GEN_TS,
  "j_per_tok_total": $ENERGY_JTOK,
  "j_per_tok_marginal": $MARGINAL_JTOK
}
JSON_EOF
fi
