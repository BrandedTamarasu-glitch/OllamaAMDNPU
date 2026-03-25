#!/usr/bin/env bash
# setup-check.sh — preflight validator for the OllamaAMDNPU XDNA2 backend
#
# Checks every prerequisite needed to run NPU-accelerated inference.
# At the end, prints a summary of what passed and what still needs doing.
#
# Usage:
#   bash tools/setup-check.sh
#
# Run from the repo root (the directory containing build/).

set -uo pipefail

PASS=0
FAIL=0
WARN=0
TODOS=()

# ── helpers ──────────────────────────────────────────────────────────────────

green()  { printf '\033[32m✔\033[0m  %s\n' "$*"; }
red()    { printf '\033[31m✘\033[0m  %s\n' "$*"; }
yellow() { printf '\033[33m⚠\033[0m  %s\n' "$*"; }
header() { printf '\n\033[1m%s\033[0m\n' "$*"; }

ok()   { green  "$1"; (( PASS++ )) || true; }
fail() { red    "$1"; (( FAIL++ )) || true; TODOS+=("$2"); }
warn() { yellow "$1"; (( WARN++ )) || true; TODOS+=("$2"); }

# ── 1. Hardware / kernel ──────────────────────────────────────────────────────

header "1. Hardware & kernel driver"

if [[ -e /dev/accel/accel0 ]]; then
    ok "/dev/accel/accel0 present (amdxdna driver loaded)"
else
    fail "/dev/accel/accel0 not found — amdxdna driver is not loaded or hardware not detected" \
         "Load amdxdna driver: kernel 6.11+ required (CachyOS 6.19 includes it). Check 'dmesg | grep xdna'."
fi

if [[ -e /dev/accel/accel0 ]]; then
    if [[ -r /dev/accel/accel0 ]]; then
        ok "Current user can read /dev/accel/accel0"
    else
        fail "Cannot access /dev/accel/accel0 — permission denied" \
             "Add yourself to the 'video' group: sudo usermod -aG video \$USER, then log out and back in."
    fi
fi

# ── 2. XRT ───────────────────────────────────────────────────────────────────

header "2. XRT (Xilinx Runtime)"

XRT_OK=0
if command -v xrt-smi &>/dev/null; then
    XRT_VER=$(xrt-smi examine 2>/dev/null | grep -A5 '^XRT$' | awk '/Version/ { print $NF; exit }') && [[ -n "$XRT_VER" ]] || XRT_VER="unknown"
    ok "xrt-smi found (version: $XRT_VER)"
    XRT_OK=1
elif command -v xbutil &>/dev/null; then
    ok "xbutil found (XRT installed)"
    XRT_OK=1
else
    fail "XRT not found — xrt-smi / xbutil not in PATH" \
         "Install XRT: 'yay -S xrt-bin' (AUR) or download from github.com/Xilinx/XRT/releases. Minimum version: 2.21.75."
fi

if [[ $XRT_OK -eq 1 ]]; then
    if ldconfig -p 2>/dev/null | grep -q libxrt_coreutil; then
        ok "libxrt_coreutil.so found in ldconfig cache"
    else
        warn "libxrt_coreutil.so not in ldconfig cache — build may fail or runtime will fail" \
             "Run 'sudo ldconfig' after XRT install, or add XRT lib path to /etc/ld.so.conf.d/."
    fi
fi

# ── 3. memlock ───────────────────────────────────────────────────────────────

header "3. Locked memory limit (required for XRT DMA buffers)"

HARD_MEMLOCK=$(ulimit -Hl 2>/dev/null || echo "unknown")
SOFT_MEMLOCK=$(ulimit -Sl 2>/dev/null || echo "unknown")

if [[ "$HARD_MEMLOCK" == "unlimited" && "$SOFT_MEMLOCK" == "unlimited" ]]; then
    ok "memlock is unlimited (hard: $HARD_MEMLOCK, soft: $SOFT_MEMLOCK)"
else
    fail "memlock is NOT unlimited (hard: $HARD_MEMLOCK, soft: $SOFT_MEMLOCK) — XRT DMA buffer pinning will fail" \
         $'Set memlock to unlimited:\n    sudo mkdir -p /etc/systemd/system/user@.service.d\n    sudo tee /etc/systemd/system/user@.service.d/memlock.conf <<EOF\n[Service]\nLimitMEMLOCK=infinity\nEOF\n    sudo systemctl daemon-reload\n  Then log out and back in. Verify with: ulimit -Hl'
fi

# ── 4. Build output ──────────────────────────────────────────────────────────

header "4. llama.cpp build"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_BENCH="$REPO_ROOT/build/bin/llama-bench"
LLAMA_CLI="$REPO_ROOT/build/bin/llama-cli"
XDNA_SO="$REPO_ROOT/build/bin/libggml-xdna.so"
VULKAN_SO="$REPO_ROOT/build/bin/libggml-vulkan.so"

if [[ -x "$LLAMA_CLI" && -x "$LLAMA_BENCH" ]]; then
    ok "llama-cli and llama-bench present in build/bin/"
else
    fail "llama-cli or llama-bench not found — llama.cpp is not built yet" \
         "Build with: cmake -B build -DGGML_XDNA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON && cmake --build build --parallel"
fi

if [[ -f "$XDNA_SO" ]]; then
    ok "libggml-xdna.so present"
elif [[ -f "${XDNA_SO}.hidden" ]]; then
    warn "libggml-xdna.so is renamed to .hidden (from a previous benchmark run)" \
         "Restore it: mv build/bin/libggml-xdna.so.hidden build/bin/libggml-xdna.so"
else
    fail "libggml-xdna.so not found — XDNA backend not built" \
         "Rebuild with -DGGML_XDNA=ON (see above)."
fi

if [[ -f "$VULKAN_SO" ]]; then
    ok "libggml-vulkan.so present"
elif [[ -f "${VULKAN_SO}.hidden" ]]; then
    warn "libggml-vulkan.so is renamed to .hidden (from a previous benchmark run)" \
         "Restore it: mv build/bin/libggml-vulkan.so.hidden build/bin/libggml-vulkan.so"
else
    warn "libggml-vulkan.so not found — Vulkan backend unavailable (NPU still works)" \
         "To enable Vulkan: rebuild with -DGGML_VULKAN=ON."
fi

# ── 5. Environment variables ─────────────────────────────────────────────────

header "5. Environment variables"

BACKEND_PATH_OK=0
if [[ -n "${GGML_BACKEND_PATH:-}" ]]; then
    if [[ -f "$GGML_BACKEND_PATH" ]]; then
        ok "GGML_BACKEND_PATH=$GGML_BACKEND_PATH"
        BACKEND_PATH_OK=1
    else
        fail "GGML_BACKEND_PATH is set but file not found: $GGML_BACKEND_PATH" \
             "Correct the path in your shell profile (~/.zshrc or ~/.bashrc). Run 'source tools/env-template.sh' to see the template."
    fi
else
    fail "GGML_BACKEND_PATH is not set — XDNA backend will not load" \
         "Source tools/env-template.sh, fill in your paths, and add to ~/.zshrc or ~/.bashrc."
fi

# Check slots
SLOTS_FOUND=0
for slot in "" "_2" "_3" "_4"; do
    xclbin_var="GGML_XDNA_XCLBIN_PATH${slot}"
    instr_var="GGML_XDNA_INSTR_PATH${slot}"
    k_var="GGML_XDNA_TILE_K${slot}"

    xclbin_val="${!xclbin_var:-}"
    instr_val="${!instr_var:-}"
    k_val="${!k_var:-}"

    slot_label="slot$([[ -z $slot ]] && echo 1 || echo ${slot#_})"

    if [[ -z "$xclbin_val" ]]; then
        if [[ -z "$slot" ]]; then
            fail "No xclbin configured (GGML_XDNA_XCLBIN_PATH not set) — NPU cannot run" \
                 "Build xclbins with mlir-aie and set GGML_XDNA_XCLBIN_PATH. See the wiki Setup Guide."
        fi
        continue
    fi

    (( SLOTS_FOUND++ )) || true
    ERR=0

    if [[ ! -f "$xclbin_val" ]]; then
        fail "$slot_label: XCLBIN file not found: $xclbin_val" \
             "Build the xclbin for K=${k_val:-?} or correct the path."
        ERR=1
    else
        ok "$slot_label (K=${k_val:-?}): xclbin found — $xclbin_val"
    fi

    if [[ -n "$instr_val" && ! -f "$instr_val" ]]; then
        fail "$slot_label: instruction file not found: $instr_val" \
             "The _sequence.bin file must be in the same directory as the xclbin."
        ERR=1
    elif [[ -n "$instr_val" ]]; then
        ok "$slot_label: instruction file found"
    fi
done

if [[ $SLOTS_FOUND -eq 0 ]]; then
    : # already reported above
elif [[ $SLOTS_FOUND -eq 1 ]]; then
    warn "Only 1 xclbin slot configured — layers with other K dimensions will fall to CPU" \
         "Add slots 2–4 for K=4096, K=5632, K=14336 to cover all LLaMA-3 8B layer types. See wiki Setup Guide."
else
    ok "$SLOTS_FOUND xclbin slots configured"
fi

# ── 6. Summary ───────────────────────────────────────────────────────────────

printf '\n%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf '\033[1mSummary:\033[0m  %d passed  |  %d warnings  |  %d failed\n' "$PASS" "$WARN" "$FAIL"
printf '%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $FAIL -eq 0 && $WARN -eq 0 ]]; then
    printf '\n\033[32mAll checks passed — you are ready to run NPU inference.\033[0m\n\n'
    printf 'Quick start:\n'
    printf '  ./build/bin/llama-cli -m models/<your-model>.gguf -p "Hello" -n 64\n\n'
    exit 0
fi

if [[ ${#TODOS[@]} -gt 0 ]]; then
    printf '\n\033[1mWhat to do next:\033[0m\n\n'
    N=1
    for todo in "${TODOS[@]}"; do
        printf '  %d. %s\n\n' "$N" "$todo"
        (( N++ )) || true
    done
fi

if [[ $FAIL -gt 0 ]]; then
    printf '\033[31mSetup is not complete — fix the items marked ✘ above before running inference.\033[0m\n\n'
    exit 1
else
    printf '\033[33mSetup has warnings — inference may work but check the ⚠ items above.\033[0m\n\n'
    exit 0
fi
