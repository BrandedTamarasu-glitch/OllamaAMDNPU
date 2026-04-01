#!/usr/bin/env bash
# validate-prefill-env.sh — Validate all 8 xclbin+insts slots in ~/.npu-prefill.env
#
# Usage: source ~/.npu-prefill.env && ./tools/validate-prefill-env.sh
#   OR:  ./tools/validate-prefill-env.sh  (auto-sources ~/.npu-prefill.env)
#
# Checks per slot:
#   1. Both XCLBIN_PATH and INSTR_PATH are set
#   2. realpath resolves (no dangling symlinks)
#   3. File exists and is non-empty
#   4. Insts file size is a multiple of 4 bytes (XRT alignment requirement)
#
# Exit code: 0 = all checks pass, 1 = any failure

set -uo pipefail

ENV_FILE="${NPU_PREFILL_ENV:-$HOME/.npu-prefill.env}"

# Always source the env file to ensure all 8 slots are loaded fresh
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

PASS=0
FAIL=0

check_slot() {
    local slot_label="$1"
    local xclbin_var="$2"
    local instr_var="$3"

    local xclbin_path="${!xclbin_var:-}"
    local instr_path="${!instr_var:-}"

    printf "  %-8s " "$slot_label"

    # Both vars must be set
    if [[ -z "$xclbin_path" || -z "$instr_path" ]]; then
        echo "SKIP (not configured)"
        return
    fi

    local errors=()

    # realpath + existence + non-empty: xclbin
    local real_xclbin
    real_xclbin=$(realpath "$xclbin_path" 2>/dev/null) || real_xclbin=""
    if [[ -z "$real_xclbin" || ! -s "$real_xclbin" ]]; then
        errors+=("xclbin missing/empty: $xclbin_path")
    fi

    # realpath + existence + non-empty: insts
    local real_instr
    real_instr=$(realpath "$instr_path" 2>/dev/null) || real_instr=""
    if [[ -z "$real_instr" || ! -s "$real_instr" ]]; then
        errors+=("insts missing/empty: $instr_path")
    else
        # 4-byte alignment check on insts file
        local instr_size
        instr_size=$(stat -c%s "$real_instr" 2>/dev/null || echo 0)
        if (( instr_size % 4 != 0 )); then
            errors+=("insts size ${instr_size} not 4-byte aligned: $instr_path")
        fi
    fi

    if [[ ${#errors[@]} -eq 0 ]]; then
        local xclbin_kb=$(( $(stat -c%s "$real_xclbin") / 1024 ))
        local instr_sz
        instr_sz=$(stat -c%s "$real_instr")
        printf "OK  xclbin=%dK  insts=%db\n" "$xclbin_kb" "$instr_sz"
        PASS=$(( PASS + 1 ))
    else
        echo "FAIL"
        for e in "${errors[@]}"; do
            echo "           ! $e"
        done
        FAIL=$(( FAIL + 1 ))
    fi
}

echo "=== validate-prefill-env ==="
echo "  Env: $ENV_FILE"
echo ""
echo "  Prefill slots (tile_n=128):"
check_slot "slot-1" "GGML_XDNA_XCLBIN_PATH"   "GGML_XDNA_INSTR_PATH"
check_slot "slot-2" "GGML_XDNA_XCLBIN_PATH_2"  "GGML_XDNA_INSTR_PATH_2"
check_slot "slot-3" "GGML_XDNA_XCLBIN_PATH_3"  "GGML_XDNA_INSTR_PATH_3"
check_slot "slot-4" "GGML_XDNA_XCLBIN_PATH_4"  "GGML_XDNA_INSTR_PATH_4"
echo ""
echo "  Decode slots (tile_n=64):"
check_slot "slot-5" "GGML_XDNA_XCLBIN_PATH_5"  "GGML_XDNA_INSTR_PATH_5"
check_slot "slot-6" "GGML_XDNA_XCLBIN_PATH_6"  "GGML_XDNA_INSTR_PATH_6"
check_slot "slot-7" "GGML_XDNA_XCLBIN_PATH_7"  "GGML_XDNA_INSTR_PATH_7"
check_slot "slot-8" "GGML_XDNA_XCLBIN_PATH_8"  "GGML_XDNA_INSTR_PATH_8"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo "=== PASS: ${PASS}/8 slots valid ==="
    exit 0
else
    echo "=== FAIL: ${FAIL} slot(s) invalid, ${PASS} OK ==="
    exit 1
fi
