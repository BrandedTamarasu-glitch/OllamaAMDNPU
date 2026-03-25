#!/usr/bin/env bash
# bench-context.sh — benchmark NPU, Vulkan, and CPU prefill across context lengths
# Usage: bench-context.sh

set -euo pipefail

LLAMA_BENCH=~/Claude/OllamaAMD/llama.cpp/build/bin/llama-bench
MODEL=~/Claude/OllamaAMD/llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
BUILD_BIN=~/Claude/OllamaAMD/llama.cpp/build/bin

VULKAN_SO=$BUILD_BIN/libggml-vulkan.so
XDNA_SO=$BUILD_BIN/libggml-xdna.so

bench() {
    local label=$1; shift
    echo "=== $label ==="
    for pp in 512 2048 4096 8192; do
        printf "  pp=%s: " $pp
        "$LLAMA_BENCH" -m "$MODEL" -p $pp -n 1 -r 1 --no-warmup "$@" 2>/dev/null \
            | awk -F'|' '/pp[0-9]/ { t=$(NF-1); gsub(/^ +/,"",t); print t+0 " t/s" }'
    done
}

cleanup() {
    [[ -f ${VULKAN_SO}.hidden ]] && mv ${VULKAN_SO}.hidden $VULKAN_SO
    [[ -f ${XDNA_SO}.hidden   ]] && mv ${XDNA_SO}.hidden   $XDNA_SO
}
trap cleanup EXIT

# CPU only (hide both backends)
mv $VULKAN_SO ${VULKAN_SO}.hidden
mv $XDNA_SO   ${XDNA_SO}.hidden
bench "CPU only"
mv ${VULKAN_SO}.hidden $VULKAN_SO
mv ${XDNA_SO}.hidden   $XDNA_SO

# NPU only (hide Vulkan, default ubatch=512)
mv $VULKAN_SO ${VULKAN_SO}.hidden
bench "NPU ub=512"
bench "NPU ub=2048" -ub 2048
mv ${VULKAN_SO}.hidden $VULKAN_SO

# Vulkan only (hide XDNA)
mv $XDNA_SO ${XDNA_SO}.hidden
bench "Vulkan" -ngl 99
mv ${XDNA_SO}.hidden $XDNA_SO
