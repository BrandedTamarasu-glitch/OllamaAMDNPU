#!/usr/bin/env bash
# build-trace-xclbin.sh — Build trace-enabled xclbin for AIE profiling
#
# Usage: ./tools/trace/build-trace-xclbin.sh [--trace-size 8192] [--tile-m 128] [--tile-k 128] [--tile-n 16]
#
# Produces: ~/xclbin-decode-trace/<M>x<K>x<N>-trace-v1.xclbin
#           ~/xclbin-decode-trace/<M>x<K>x<N>-trace-v1_insts.txt
#
# WARNING: The _insts.txt from a trace build is NOT compatible with
# a non-trace xclbin. Never mix trace/non-trace instruction files.

set -euo pipefail

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/Claude/OllamaAMD/mlir-aie}"
WHOLE_ARRAY_DIR="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"
OUTPUT_DIR="${XDNA_TRACE_XCLBIN_DIR:-$HOME/xclbin-decode-trace}"

# Defaults matching current decode config
TRACE_SIZE=8192
TILE_M=128
TILE_K=128
TILE_N=16

# Parse named args
while [[ $# -gt 0 ]]; do
    case $1 in
        --trace-size) TRACE_SIZE="$2"; shift 2 ;;
        --tile-m) TILE_M="$2"; shift 2 ;;
        --tile-k) TILE_K="$2"; shift 2 ;;
        --tile-n) TILE_N="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

XCLBIN_NAME="${TILE_M}x${TILE_K}x${TILE_N}-trace-v1"

echo "=== Building trace-enabled xclbin ==="
echo "  Tiles: ${TILE_M}x${TILE_K}x${TILE_N}"
echo "  Trace buffer: ${TRACE_SIZE} bytes"
echo "  Output: ${OUTPUT_DIR}/${XCLBIN_NAME}.xclbin"

# Validate source exists
if [[ ! -f "$WHOLE_ARRAY_DIR/whole_array.py" ]]; then
    echo "ERROR: whole_array.py not found at $WHOLE_ARRAY_DIR/"
    exit 1
fi

# Build using existing makefile trace target (makefile-common line 199)
cd "$WHOLE_ARRAY_DIR"
make trace \
    M="$TILE_M" K="$TILE_K" N="$TILE_N" \
    m="$TILE_M" k="$TILE_K" n="$TILE_N" \
    trace_size="$TRACE_SIZE" 2>&1

# Locate output (build dir may vary)
BUILD_DIR="$WHOLE_ARRAY_DIR/build"
if [[ ! -f "$BUILD_DIR/final.xclbin" ]]; then
    echo "ERROR: Build did not produce final.xclbin in $BUILD_DIR/"
    echo "Check makefile-common trace target or run manually:"
    echo "  python3 whole_array.py --dev npu2 -M $TILE_M -K $TILE_K -N $TILE_N --trace_size $TRACE_SIZE"
    exit 1
fi

# Copy to output
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/final.xclbin" "$OUTPUT_DIR/${XCLBIN_NAME}.xclbin"
cp "$BUILD_DIR/insts.txt" "$OUTPUT_DIR/${XCLBIN_NAME}_insts.txt"

# Write manifest entry
MANIFEST="$OUTPUT_DIR/${XCLBIN_NAME}-manifest.json"
KERNEL_HASH=$(cd "$MLIR_AIE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
cat > "$MANIFEST" <<MANIFEST_EOF
{
  "xclbin_path": "$OUTPUT_DIR/${XCLBIN_NAME}.xclbin",
  "kernel_source": "mm.cc",
  "kernel_source_hash": "git:${KERNEL_HASH}",
  "compiler": "peano",
  "compiler_flags": "-M $TILE_M -K $TILE_K -N $TILE_N --trace_size $TRACE_SIZE",
  "fifo_depth": 2,
  "c_row_maj": true,
  "opt_perf_enabled": false,
  "trace_enabled": true,
  "build_date": "$(date -Iseconds)",
  "build_commit": "git:${KERNEL_HASH}",
  "license": "apache-2.0",
  "notes": "Standalone trace xclbin for Phase 13A profiling"
}
MANIFEST_EOF

echo "=== Build complete ==="
echo "  xclbin: $OUTPUT_DIR/${XCLBIN_NAME}.xclbin"
echo "  instr:  $OUTPUT_DIR/${XCLBIN_NAME}_insts.txt"
echo "  manifest: $MANIFEST"
echo ""
echo "WARNING: This _insts.txt is NOT compatible with non-trace xclbins."
