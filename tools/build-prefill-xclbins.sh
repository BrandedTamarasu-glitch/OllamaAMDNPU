#!/usr/bin/env bash
# build-prefill-xclbins.sh — Build tile_n=128 prefill xclbins for Phase 17B
#
# Usage: ./tools/build-prefill-xclbins.sh [OPTIONS]
#
# Builds 4 xclbins (K=2048/4096/5632/14336) with M=128, N=128 into a staging
# directory, then atomically promotes to ~/xclbin-prefill/ if ≥2/4 succeed.
#
# Inner tiles: m=32 k=128 n=32 (constrained by M=128/N=128 with 4 AIE rows/cols)
# Required env: mlir-aie-env venv + PEANO_INSTALL_DIR
#
# Options:
#   --output-dir DIR    Destination dir (default: ~/xclbin-prefill)
#   --tile-m N          Override TILE_M (default: 128)
#   --k-values "A B C"  Space-separated K values (default: "2048 4096 5632 14336")
#   --dry-run           Print commands without executing
#   --skip-build        Skip make; only write manifest from existing staging files
#
# Outputs per K:
#   ~/xclbin-prefill/k${K}_n128_prefill.xclbin
#   ~/xclbin-prefill/k${K}_n128_prefill.txt
#   ~/xclbin-prefill/manifest.json

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
OUTPUT_DIR="$HOME/xclbin-prefill"
STAGING_DIR="$HOME/xclbin-prefill-staging"
TILE_M=128
TILE_N=128
INNER_M=32
INNER_K=128
INNER_N=32
N_AIE_COLS=4
K_VALUES=(2048 4096 5632 14336)
DRY_RUN=0
SKIP_BUILD=0

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/Claude/OllamaAMD/mlir-aie}"
VENV_DIR="$HOME/Claude/OllamaAMD/mlir-aie-env"
WHOLE_ARRAY_DIR="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)   OUTPUT_DIR="$2";       shift 2 ;;
        --tile-m)       TILE_M="$2";           shift 2 ;;
        --k-values)     read -ra K_VALUES <<< "$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1;             shift ;;
        --skip-build)   SKIP_BUILD=1;          shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

elapsed() {
    local start=$1 end=$2
    local secs=$(( end - start ))
    printf "%dm %ds" $(( secs / 60 )) $(( secs % 60 ))
}

# ── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: mlir-aie venv not found at $VENV_DIR"
    exit 1
fi

if [[ ! -f "$WHOLE_ARRAY_DIR/whole_array.py" ]]; then
    echo "ERROR: whole_array.py not found at $WHOLE_ARRAY_DIR/"
    exit 1
fi

# ── Environment setup ────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 && $SKIP_BUILD -eq 0 ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    PEANO_INSTALL_DIR="$VENV_DIR/lib/python3.14/site-packages/llvm-aie"
    if [[ ! -f "$PEANO_INSTALL_DIR/bin/clang++" ]]; then
        # Fallback: search for any python3.x version in the venv
        PEANO_INSTALL_DIR="$(find "$VENV_DIR/lib" -name "llvm-aie" -type d 2>/dev/null | head -1)"
        if [[ -z "$PEANO_INSTALL_DIR" ]]; then
            echo "ERROR: Peano (llvm-aie) not found in $VENV_DIR"
            exit 1
        fi
    fi
    export PEANO_INSTALL_DIR
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== build-prefill-xclbins.sh ==="
echo "  TILE_M=${TILE_M}  TILE_N=${TILE_N}"
echo "  Inner: m=${INNER_M} k=${INNER_K} n=${INNER_N}  cols=${N_AIE_COLS}"
echo "  K values: ${K_VALUES[*]}"
echo "  Output: ${OUTPUT_DIR}"
[[ $DRY_RUN -eq 1 ]]   && echo "  [DRY-RUN MODE]"
[[ $SKIP_BUILD -eq 1 ]] && echo "  [SKIP-BUILD MODE]"
echo ""

# ── Staging setup ────────────────────────────────────────────────────────────
run mkdir -p "$STAGING_DIR"

# ── Build loop ───────────────────────────────────────────────────────────────
SUCCEEDED=0
FAILED=0
IDX=0
declare -A BUILD_STATUS

for K in "${K_VALUES[@]}"; do
    IDX=$(( IDX + 1 ))
    TOTAL=${#K_VALUES[@]}
    LOG="$STAGING_DIR/build-k${K}.log"

    # Expected output filenames from make all (parameterised suffix)
    TARGET_SUFFIX="${TILE_M}x${K}x${TILE_N}_${INNER_M}x${INNER_K}x${INNER_N}_${N_AIE_COLS}c"
    SRC_XCLBIN="$WHOLE_ARRAY_DIR/build/final_${TARGET_SUFFIX}.xclbin"
    SRC_INSTS="$WHOLE_ARRAY_DIR/build/insts_${TARGET_SUFFIX}.txt"

    # Destination filenames (clean k${K}_n${TILE_N}_prefill.* convention)
    DST_XCLBIN="$STAGING_DIR/k${K}_n${TILE_N}_prefill.xclbin"
    DST_INSTS="$STAGING_DIR/k${K}_n${TILE_N}_prefill.txt"

    printf "[%d/%d] Building K=%-5d (tile_n=%d)... " "$IDX" "$TOTAL" "$K" "$TILE_N"
    BUILD_START=$(date +%s)

    if [[ $SKIP_BUILD -eq 1 ]]; then
        # Validate existing staging files instead of building
        if [[ -s "$DST_XCLBIN" && -s "$DST_INSTS" ]]; then
            echo "skipped (staging files present)"
            BUILD_STATUS[$K]="ok"
            SUCCEEDED=$(( SUCCEEDED + 1 ))
        else
            echo "FAILED (staging files missing, use without --skip-build)"
            BUILD_STATUS[$K]="missing"
            FAILED=$(( FAILED + 1 ))
        fi
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] make all M=${TILE_M} K=${K} N=${TILE_N} m=${INNER_M} k=${INNER_K} n=${INNER_N} NPU2=1"
        BUILD_STATUS[$K]="dry-run"
        continue
    fi

    # Remove stale MLIR to force npu2 regeneration
    rm -f "$WHOLE_ARRAY_DIR/build/aie_${TARGET_SUFFIX}.mlir"

    BUILD_OK=1
    (
        cd "$WHOLE_ARRAY_DIR"
        make all \
            M="$TILE_M" K="$K" N="$TILE_N" \
            m="$INNER_M" k="$INNER_K" n="$INNER_N" \
            n_aie_cols="$N_AIE_COLS" \
            NPU2=1
    ) > "$LOG" 2>&1 || BUILD_OK=0

    BUILD_END=$(date +%s)

    if [[ $BUILD_OK -eq 1 && -s "$SRC_XCLBIN" && -s "$SRC_INSTS" ]]; then
        cp "$SRC_XCLBIN" "$DST_XCLBIN"
        cp "$SRC_INSTS"  "$DST_INSTS"
        echo "done in $(elapsed "$BUILD_START" "$BUILD_END")"
        BUILD_STATUS[$K]="ok"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    else
        echo "FAILED (see $LOG)"
        BUILD_STATUS[$K]="failed"
        FAILED=$(( FAILED + 1 ))
    fi
done

# ── Promotion gate ────────────────────────────────────────────────────────────
echo ""
echo "=== Results: ${SUCCEEDED}/${TOTAL} succeeded ==="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would promote staging → ${OUTPUT_DIR} if ≥2 succeeded"
    exit 0
fi

if [[ $SUCCEEDED -lt 2 ]]; then
    echo "ERROR: Only ${SUCCEEDED}/4 builds succeeded — minimum 2 required for promotion."
    echo "       Staging dir preserved at: ${STAGING_DIR}"
    echo "       Per-build logs:"
    for K in "${K_VALUES[@]}"; do
        printf "         K=%-5d  %s\n" "$K" "${BUILD_STATUS[$K]:-unknown}"
    done
    exit 1
fi

# ── Atomic promotion ──────────────────────────────────────────────────────────
echo "Promoting staging → ${OUTPUT_DIR}"

# Backup existing output dir if present
if [[ -d "$OUTPUT_DIR" ]]; then
    BACKUP="${OUTPUT_DIR}.bak.$(date +%s)"
    mv "$OUTPUT_DIR" "$BACKUP"
    echo "  (previous dir backed up to $BACKUP)"
fi

mv "$STAGING_DIR" "$OUTPUT_DIR"

# ── Manifest ──────────────────────────────────────────────────────────────────
MLIR_AIE_COMMIT=$(cd "$MLIR_AIE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(date -Iseconds)

MANIFEST="$OUTPUT_DIR/manifest.json"
{
    echo "{"
    echo "  \"build_date\": \"${BUILD_DATE}\","
    echo "  \"mlir_aie_commit\": \"${MLIR_AIE_COMMIT}\","
    echo "  \"tile_m\": ${TILE_M},"
    echo "  \"tile_n\": ${TILE_N},"
    echo "  \"inner_m\": ${INNER_M},"
    echo "  \"inner_k\": ${INNER_K},"
    echo "  \"inner_n\": ${INNER_N},"
    echo "  \"n_aie_cols\": ${N_AIE_COLS},"
    echo "  \"device\": \"npu2\","
    echo "  \"k_values\": {"
    FIRST=1
    for K in "${K_VALUES[@]}"; do
        [[ $FIRST -eq 0 ]] && echo ","
        STATUS="${BUILD_STATUS[$K]:-unknown}"
        XCLBIN_FILE="$OUTPUT_DIR/k${K}_n${TILE_N}_prefill.xclbin"
        XCLBIN_SIZE=$(stat -c%s "$XCLBIN_FILE" 2>/dev/null || echo 0)
        printf "    \"k%s\": {\"status\": \"%s\", \"xclbin_bytes\": %s}" \
            "$K" "$STATUS" "$XCLBIN_SIZE"
        FIRST=0
    done
    echo ""
    echo "  }"
    echo "}"
} > "$MANIFEST"

echo ""
echo "=== Done ==="
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Manifest:     ${MANIFEST}"
echo "  Builds:"
for K in "${K_VALUES[@]}"; do
    STATUS="${BUILD_STATUS[$K]:-unknown}"
    if [[ "$STATUS" == "ok" ]]; then
        printf "    k%-5d  %-6s  %s\n" "$K" "$STATUS" "$(ls -sh "$OUTPUT_DIR/k${K}_n${TILE_N}_prefill.xclbin" 2>/dev/null | awk '{print $1}')"
    else
        printf "    k%-5d  %s\n" "$K" "$STATUS"
    fi
done
