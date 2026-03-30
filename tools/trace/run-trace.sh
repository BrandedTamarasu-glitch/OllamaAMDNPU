#!/usr/bin/env bash
# run-trace.sh — Run standalone trace capture via whole_array.exe
#
# Usage: ./tools/trace/run-trace.sh [--xclbin PATH] [--output-dir PATH]
#
# Requires: trace-enabled xclbin built by build-trace-xclbin.sh

set -euo pipefail

XCLBIN_DIR="${XDNA_TRACE_XCLBIN_DIR:-$HOME/xclbin-decode-trace}"
XCLBIN="${XCLBIN_DIR}/128x128x16-trace-v1.xclbin"
INSTR="${XCLBIN_DIR}/128x128x16-trace-v1_insts.txt"
OUTPUT_DIR="${GGML_XDNA_TRACE_DIR:-./trace-output}"
WHOLE_ARRAY_DIR="${MLIR_AIE_DIR:-$HOME/Claude/OllamaAMD/mlir-aie}/programming_examples/basic/matrix_multiplication/whole_array"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --xclbin) XCLBIN="$2"; shift 2 ;;
        --instr) INSTR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate inputs
if [[ ! -f "$XCLBIN" ]]; then
    echo "ERROR: Trace xclbin not found: $XCLBIN"
    echo "Run build-trace-xclbin.sh first."
    exit 1
fi

if [[ ! -f "$INSTR" ]]; then
    echo "ERROR: Instruction file not found: $INSTR"
    echo "Expected alongside xclbin at: $INSTR"
    exit 1
fi

# Validate output directory
mkdir -p "$OUTPUT_DIR"
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output directory not writable: $OUTPUT_DIR"
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
RAW_OUTPUT="$OUTPUT_DIR/${TIMESTAMP}-trace-raw.bin"

echo "=== Running trace capture ==="
echo "  xclbin: $XCLBIN"
echo "  instr:  $INSTR"
echo "  output: $RAW_OUTPUT"

# Run the test executable with trace xclbin
cd "$WHOLE_ARRAY_DIR"

# Find the executable (name varies by build: mat_mul_whole_array.exe or similar)
EXE=$(find build/ -name "*.exe" -type f | head -1)
if [[ -z "$EXE" ]]; then
    echo "ERROR: No .exe found in $WHOLE_ARRAY_DIR/build/"
    echo "Build the trace xclbin first with build-trace-xclbin.sh"
    exit 1
fi

echo "  Executable: $EXE"
"$EXE" -x "$XCLBIN" -i "$INSTR" -k MLIR_AIE \
    -M 128 -K 128 -N 16 -t 8192 2>&1

# The trace target writes trace data to trace.txt in cwd
if [[ -f "trace.txt" ]]; then
    mv "trace.txt" "$RAW_OUTPUT"
fi

if [[ -f "$RAW_OUTPUT" ]] && [[ -s "$RAW_OUTPUT" ]]; then
    echo "=== Trace captured ==="
    echo "  Raw output: $RAW_OUTPUT ($(stat -c%s "$RAW_OUTPUT") bytes)"
    echo "  Next: run parse-trace.sh --input $RAW_OUTPUT"
else
    echo "WARNING: Trace output is empty or missing."
    echo "Check that the xclbin has trace configuration enabled."
    exit 1
fi
