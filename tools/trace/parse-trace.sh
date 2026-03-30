#!/usr/bin/env bash
# parse-trace.sh — Convert raw AIE trace to Chrome DevTools JSON
#
# Usage: ./tools/trace/parse-trace.sh --input <raw.bin> [--mlir <source.mlir>] [--output <chrome.json>]
#
# Wraps mlir-aie's parse.py. Output opens in chrome://tracing or Perfetto UI.

set -euo pipefail

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/Claude/OllamaAMD/mlir-aie}"
PARSE_PY="$MLIR_AIE_DIR/python/utils/trace/parse.py"
WHOLE_ARRAY_DIR="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"

INPUT=""
MLIR_SOURCE=""  # Required — parse.py --mlir is mandatory
OUTPUT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --mlir) MLIR_SOURCE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$INPUT" ]]; then
    echo "ERROR: --input <raw-trace.bin> required"
    exit 1
fi

if [[ -z "$MLIR_SOURCE" ]]; then
    echo "ERROR: --mlir <source.mlir> required (parse.py requires it)"
    echo "Typically: build/aie_trace.mlir in the build directory"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

if [[ ! -s "$INPUT" ]]; then
    echo "ERROR: Input file is empty: $INPUT"
    exit 1
fi

if [[ ! -f "$PARSE_PY" ]]; then
    echo "ERROR: parse.py not found: $PARSE_PY"
    echo "Check MLIR_AIE_DIR environment variable."
    exit 1
fi

# Default output: same name as input but .json
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${INPUT%.bin}.json"
fi

echo "=== Parsing trace ==="
echo "  Input:  $INPUT"
echo "  MLIR:   $MLIR_SOURCE"
echo "  Output: $OUTPUT"

python3 "$PARSE_PY" --input "$INPUT" --mlir "$MLIR_SOURCE" --output "$OUTPUT"

if [[ -f "$OUTPUT" ]] && [[ -s "$OUTPUT" ]]; then
    echo "=== Parse complete ==="
    echo "  Chrome DevTools JSON: $OUTPUT"
    echo "  Open in: chrome://tracing or https://ui.perfetto.dev/"
else
    echo "ERROR: Parse produced no output."
    exit 1
fi
