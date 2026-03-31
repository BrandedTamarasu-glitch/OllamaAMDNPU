#!/usr/bin/env bash
# sweep.sh — Automated parameter matrix sweep for AIE kernel optimisation
#
# Usage: ./tools/sweep/sweep.sh [OPTIONS]
#
# Options:
#   --output FILE     CSV output path (default: tools/sweep/sweep-results.csv)
#   --model  FILE     GGUF model path (default: $MODEL env var or hardcoded path)
#   --skip-build      Skip xclbin rebuild; use pre-built xclbins in ~/xclbin-sweep/
#   --skip-fifo       Fix fifo_depth=2 (use when trace shows instruction-limited)
#
# Sweeps: fifo_depth {2,3,4} × c_col_maj {0,1} × opt_perf {0,1} = 12 combos
# Each combo: rebuild xclbin (~10 min), run llama-bench, record t/s.
# Full matrix: ~2 hours. Safe to leave running overnight.
#
# Outer matrix dims (M×K×N) must match model layer dimensions so that GGML
# dispatches ops to the XDNA backend.  Defaults target Llama-3.x-8B decode:
#   K=14336 (FFN down-projection)
# Adjust M_OUTER / K_OUTER / N_OUTER via env vars to target different layers.
#
# Inner tile dims (m×k×n) are fixed at 128×128×16 — the proven decode tile.

set -euo pipefail

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/Claude/OllamaAMD/mlir-aie}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Claude/OllamaAMD/llama.cpp}"
WHOLE_ARRAY_DIR="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"
VENV="$HOME/Claude/OllamaAMD/mlir-aie-env/bin/activate"

# Default model — prefer Q8_0 for accurate decode benchmark
MODEL="${MODEL:-$HOME/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf}"

# Outer matrix dims: match the model layer you want to benchmark
M_OUTER="${M_OUTER:-2048}"
K_OUTER="${K_OUTER:-14336}"
N_OUTER="${N_OUTER:-64}"

# Inner tile dims (fixed; changing these requires a different kernel binary)
TILE_M=128
TILE_K=128
TILE_N=16
N_AIE_COLS=4

OUTPUT_CSV="tools/sweep/sweep-results.csv"
SKIP_BUILD=0
SKIP_FIFO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)    OUTPUT_CSV="$2"; shift 2 ;;
        --model)     MODEL="$2";      shift 2 ;;
        --skip-build) SKIP_BUILD=1;   shift ;;
        --skip-fifo)  SKIP_FIFO=1;    shift ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL" >&2
    echo "       Set MODEL env var or use --model" >&2
    exit 1
fi
if [[ ! -d "$WHOLE_ARRAY_DIR" ]]; then
    echo "ERROR: mlir-aie whole_array dir not found: $WHOLE_ARRAY_DIR" >&2
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: venv not found: $VENV" >&2
    exit 1
fi
source "$VENV"

# ── Sweep parameters ──────────────────────────────────────────────────────────
if [[ $SKIP_FIFO -eq 1 ]]; then
    FIFO_VALUES=(2)
    echo "NOTE: --skip-fifo — fifo_depth fixed at 2 (instruction-limited trace result)"
else
    FIFO_VALUES=(2 3 4)
fi
CROW_VALUES=(0 1)
OPT_VALUES=(0 1)

TOTAL=$(( ${#FIFO_VALUES[@]} * ${#CROW_VALUES[@]} * ${#OPT_VALUES[@]} ))

# Derived xclbin naming: final_${M}x${K}x${N}_${m}x${k}x${n}_${n}c.xclbin
TARGET_SUFFIX="${M_OUTER}x${K_OUTER}x${N_OUTER}_${TILE_M}x${TILE_K}x${TILE_N}_${N_AIE_COLS}c"

# ── CSV header ────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT_CSV")"
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "timestamp,fifo_depth,c_col_maj,opt_perf,M,K,N,m,k,n,n_aie_cols,xclbin_hash,decode_ts,notes" \
        > "$OUTPUT_CSV"
fi

echo "=== Phase 14A Parameter Sweep ==="
echo "  Outer dims : ${M_OUTER}×${K_OUTER}×${N_OUTER}"
echo "  Inner tile : ${TILE_M}×${TILE_K}×${TILE_N}  n_aie_cols=${N_AIE_COLS}"
echo "  Matrix     : fifo_depth {${FIFO_VALUES[*]}} × c_col_maj {0,1} × opt_perf {0,1} = $TOTAL combos"
echo "  Model      : $MODEL"
echo "  Output     : $OUTPUT_CSV"
echo ""

COMBO=0

for FIFO in "${FIFO_VALUES[@]}"; do
for CROW in "${CROW_VALUES[@]}"; do
for OPT  in "${OPT_VALUES[@]}";  do

    COMBO=$((COMBO + 1))
    TIMESTAMP=$(date -Iseconds)
    XCLBIN_LABEL="f${FIFO}_c${CROW}_o${OPT}"
    echo "--- Combo $COMBO/$TOTAL  [$XCLBIN_LABEL] ---"

    XCLBIN_STORE="$HOME/xclbin-sweep/$XCLBIN_LABEL"
    XCLBIN="$XCLBIN_STORE/final_${TARGET_SUFFIX}.xclbin"
    INSTS="$XCLBIN_STORE/insts_${TARGET_SUFFIX}.txt"

    # ── Build ──────────────────────────────────────────────────────────────
    if [[ $SKIP_BUILD -eq 0 ]] || [[ ! -f "$XCLBIN" ]]; then
        echo "  Building xclbin (fifo_depth=$FIFO c_col_maj=$CROW opt_perf=$OPT)..."
        mkdir -p "$XCLBIN_STORE"

        # fifo_depth has no Makefile variable — patch whole_array.py directly.
        # sed is idempotent: subsequent runs always land on $FIFO.
        sed -i "s/fifo_depth=[0-9]*/fifo_depth=$FIFO/" "$WHOLE_ARRAY_DIR/whole_array.py"

        make -C "$WHOLE_ARRAY_DIR" \
            NPU2=1 \
            M="$M_OUTER" K="$K_OUTER" N="$N_OUTER" \
            m="$TILE_M"  k="$TILE_K"  n="$TILE_N" \
            n_aie_cols="$N_AIE_COLS" \
            c_col_maj="$CROW" \
            opt_perf="$OPT" \
            dtype_in=i8 dtype_out=i32 \
            2>&1 | tail -8

        BUILT_XCLBIN="$WHOLE_ARRAY_DIR/build/final_${TARGET_SUFFIX}.xclbin"
        BUILT_INSTS="$WHOLE_ARRAY_DIR/build/insts_${TARGET_SUFFIX}.txt"

        if [[ ! -f "$BUILT_XCLBIN" ]]; then
            echo "  ERROR: build failed — xclbin not found"
            echo "$TIMESTAMP,$FIFO,$CROW,$OPT,$M_OUTER,$K_OUTER,$N_OUTER,$TILE_M,$TILE_K,$TILE_N,$N_AIE_COLS,,0,BUILD_FAILED" \
                >> "$OUTPUT_CSV"
            # Restore fifo_depth to default before continuing
            sed -i "s/fifo_depth=[0-9]*/fifo_depth=2/" "$WHOLE_ARRAY_DIR/whole_array.py"
            continue
        fi

        cp "$BUILT_XCLBIN" "$XCLBIN"
        cp "$BUILT_INSTS"  "$INSTS"
        echo "  Build OK → $XCLBIN_STORE/"
    else
        echo "  Skipping build — using $XCLBIN_STORE/"
    fi

    # ── Hash ───────────────────────────────────────────────────────────────
    XHASH=$(sha256sum "$XCLBIN" | cut -c1-12)

    # ── Benchmark ─────────────────────────────────────────────────────────
    echo "  Benchmarking (decode t/s)..."
    BENCH_LOG=$(mktemp)
    set +e
    GGML_VK_VISIBLE_DEVICES="" \
    GGML_XDNA_XCLBIN_PATH="$XCLBIN" \
    GGML_XDNA_INSTR_PATH="$INSTS" \
    GGML_XDNA_TILE_M="$M_OUTER" \
    GGML_XDNA_TILE_K="$K_OUTER" \
    GGML_XDNA_TILE_N="$N_OUTER" \
    GGML_XDNA_MIN_N=1 \
    GGML_XDNA_MAX_N=1 \
        "$LLAMA_CPP_DIR/build/bin/llama-bench" \
        -m "$MODEL" -p 0 -n 30 -r 3 \
        > "$BENCH_LOG" 2>&1
    BENCH_EXIT=$?
    set -e

    if [[ $BENCH_EXIT -ne 0 ]]; then
        echo "  ERROR: llama-bench exited $BENCH_EXIT"
        echo "$TIMESTAMP,$FIFO,$CROW,$OPT,$M_OUTER,$K_OUTER,$N_OUTER,$TILE_M,$TILE_K,$TILE_N,$N_AIE_COLS,$XHASH,0,BENCH_FAILED" \
            >> "$OUTPUT_CSV"
        rm -f "$BENCH_LOG"
        continue
    fi

    # Parse t/s — llama-bench markdown: | model | ... | tg30 | 43.64 ± 0.18 |
    DECODE_TS=$(grep -oP '\|\s+[\d.]+\s+±' "$BENCH_LOG" | grep -oP '[\d.]+' | head -1 || echo "0")
    echo "  decode = ${DECODE_TS} t/s"
    rm -f "$BENCH_LOG"

    echo "$TIMESTAMP,$FIFO,$CROW,$OPT,$M_OUTER,$K_OUTER,$N_OUTER,$TILE_M,$TILE_K,$TILE_N,$N_AIE_COLS,$XHASH,$DECODE_TS," \
        >> "$OUTPUT_CSV"
    echo ""

done
done
done

# Restore fifo_depth to default after sweep completes
sed -i "s/fifo_depth=[0-9]*/fifo_depth=2/" "$WHOLE_ARRAY_DIR/whole_array.py"

echo "=== Sweep complete: $COMBO/$TOTAL combos ==="
echo "Results: $OUTPUT_CSV"
echo ""
column -t -s ',' "$OUTPUT_CSV" 2>/dev/null || cat "$OUTPUT_CSV"
