#!/usr/bin/env bash
# Task 7: NPU power baseline measurement
# Uses amdgpu PPT (Package Power Tracking) via hwmon — covers CPU+GPU+NPU package
# PPT values in µW; converted to W in output

POWER_NODE=/sys/class/hwmon/hwmon3/power1_average
XCLBIN=/home/corye/Claude/OllamaAMD/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build/final_64x2048x64_64x64x64.xclbin
INSTS=/home/corye/Claude/OllamaAMD/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build/insts_64x2048x64_64x64x64.txt
TEST_BIN=/home/corye/Claude/OllamaAMD/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/_build/single_core
KERNEL=MLIR_AIE
ITERS=200
WARMUP=20
RESULT_FILE=/home/corye/Claude/OllamaAMD/llama.cpp/docs/superpowers/power-baseline.md

uw_to_w() { awk "BEGIN{printf \"%.3f\", $1/1000000}"; }

mean_uw() {
    # args: space-separated µW values
    local arr=("$@") sum=0
    for v in "${arr[@]}"; do sum=$((sum + v)); done
    echo $((sum / ${#arr[@]}))
}

echo "=== NPU Power Baseline — $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Power node: $POWER_NODE"
echo ""

# ── 1. Idle baseline: 24 samples × 0.25s = 6s ────────────────────────────────
echo "Sampling idle power (6s)..."
IDLE_SAMPLES=()
for i in $(seq 1 24); do
    IDLE_SAMPLES+=("$(cat "$POWER_NODE")")
    sleep 0.25
done
IDLE_UW=$(mean_uw "${IDLE_SAMPLES[@]}")
IDLE_W=$(uw_to_w "$IDLE_UW")
echo "  Idle: ${IDLE_W} W  (${#IDLE_SAMPLES[@]} samples)"
echo ""

# ── 2. Load NPU ───────────────────────────────────────────────────────────────
echo "Starting matmul ($ITERS iters, $WARMUP warmup)..."
source /home/corye/Claude/OllamaAMD/mlir-aie-env/bin/activate

"$TEST_BIN" \
    -x "$XCLBIN" \
    -k "$KERNEL" \
    -i "$INSTS" \
    --verify false \
    -M 64 -K 2048 -N 64 \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --verbosity 0 \
    > /tmp/matmul_run.log 2>&1 &
NPU_PID=$!

sleep 1.5
echo "  NPU PID: $NPU_PID — sampling..."

LOAD_SAMPLES=()
while kill -0 "$NPU_PID" 2>/dev/null; do
    LOAD_SAMPLES+=("$(cat "$POWER_NODE")")
    sleep 0.25
done
wait "$NPU_PID" || true

if [ ${#LOAD_SAMPLES[@]} -eq 0 ]; then
    echo "WARNING: no samples — job finished before sampling started"
    LOAD_W="N/A"
    DELTA_W="N/A"
else
    LOAD_UW=$(mean_uw "${LOAD_SAMPLES[@]}")
    LOAD_W=$(uw_to_w "$LOAD_UW")
    DELTA_W=$(awk "BEGIN{printf \"%.3f\", $LOAD_W - $IDLE_W}")
    echo "  Loaded: ${LOAD_W} W  (${#LOAD_SAMPLES[@]} samples)"
fi

echo ""
echo "--- matmul stdout ---"
cat /tmp/matmul_run.log
echo "---------------------"
echo ""

echo "=== Results ==="
printf "  Idle power  : %s W\n" "$IDLE_W"
printf "  NPU loaded  : %s W\n" "$LOAD_W"
printf "  Delta (NPU) : %s W\n" "$DELTA_W"
echo ""

# ── 3. Write results doc ──────────────────────────────────────────────────────
mkdir -p "$(dirname "$RESULT_FILE")"
cat > "$RESULT_FILE" <<EOF
# Task 7: NPU Power Baseline

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Workload:** single_core i8 matmul 64×2048×64 (tile 64×64×64), $ITERS iterations
**Measurement:** amdgpu PPT (\`$POWER_NODE\`) — µW, covers CPU+GPU+NPU package

| Condition       | Power (W) |
|-----------------|-----------|
| Idle baseline   | ${IDLE_W} |
| NPU loaded      | ${LOAD_W} |
| Delta (NPU+DMA) | ${DELTA_W} |

## Notes
- PPT = Package Power Tracking — whole SoC (CPU + GPU + NPU), no per-component isolation
- Delta includes host CPU overhead from the test binary's DMA dispatch loop
- Intel RAPL \`energy_uj\` exists at \`/sys/class/powercap/intel-rapl:0/\` but requires root
- Sample rate: 4 Hz (0.25s interval)
- Idle window: 6s (24 samples); loaded window: duration of $ITERS-iter run
EOF

echo "Written: $RESULT_FILE"
