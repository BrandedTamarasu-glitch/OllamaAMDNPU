#!/usr/bin/env bash
# env-template.sh — environment variable template for the OllamaAMDNPU XDNA2 backend
#
# HOW TO USE:
#   1. Copy the exports below into your ~/.zshrc or ~/.bashrc
#   2. Replace the placeholder paths with your actual xclbin locations
#   3. Run: source ~/.zshrc  (or open a new terminal)
#   4. Verify setup: bash tools/setup-check.sh
#
# BUILDING XCLBINS:
#   xclbins are compiled NPU kernels — one per K dimension. For LLaMA-3 8B you need
#   K=2048, K=4096, K=5632, K=14336. Build instructions are in the wiki Setup Guide:
#   https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU/wiki/Setup-Guide
#
# WHAT IS A SLOT?
#   Each "slot" is one compiled xclbin covering one K dimension. The backend picks
#   the slot whose TILE_K matches the current matmul. Matmuls with no matching slot
#   fall back to CPU automatically — this is expected behaviour.
#
# PHASE 7 MODE (NPU decode + Vulkan prefill):
#   - MIN_N=1, MAX_N=1: NPU handles only N=1 (single-token decode)
#   - All prefill ops (N>1) route to Vulkan automatically (~930 t/s pp512)
#   - NPU decode: ~0.65 t/s (DMA-limited; Phase 10 targets ~15 t/s via BO pre-staging)
#   - NOTE: Phase 7 originally claimed ~42 t/s — that was Vulkan decode (ngl=99 default).
#     See Phase 9 correction for details.
#   - xclbins: TILE_N=64 decode kernels (n_aie_cols=4)
#
# PHASE 6 MODE (NPU prefill only) — NPU handles prefill, CPU handles decode:
#   - Set MIN_N=2, MAX_N=131072 (or omit MAX_N)
#   - Use TILE_N=256 prefill xclbins (n_aie_cols=4)
#   - NPU prefill: ~19.5 t/s pp2048 vs ~4.3 t/s CPU

# ── Build location ────────────────────────────────────────────────────────────
# Adjust REPO to wherever you cloned OllamaAMDNPU
REPO=~/OllamaAMDNPU

# ── Backend selection ─────────────────────────────────────────────────────────
# Points llama.cpp to the XDNA backend shared library (loaded alongside Vulkan).
export GGML_BACKEND_PATH=$REPO/build/bin/libggml-xdna.so

# ── Routing: NPU-decode-only mode (Phase 7) ───────────────────────────────────
# MIN_N=1: include single-token decode (N=1) in NPU offload
# MAX_N=1: cap NPU at N=1, letting Vulkan handle all larger prefill batches
export GGML_XDNA_MIN_N=1
export GGML_XDNA_MAX_N=1
# To use Phase 6 NPU-prefill-only mode instead, replace the above with:
#   export GGML_XDNA_MIN_N=2
#   # (remove or unset GGML_XDNA_MAX_N)

# ── Slot 1: K=2048 (attention Q/K/V/O projections) ───────────────────────────
# Required. Build command (decode xclbin, TILE_N=64, 4-col):
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
#        M=2048 K=2048 N=64 m=64 k=64 n=64 n_aie_cols=4
export GGML_XDNA_XCLBIN_PATH=~/xclbin-decode/k2048_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH=~/xclbin-decode/k2048_n64_decode.txt
export GGML_XDNA_TILE_M=2048
export GGML_XDNA_TILE_K=2048
export GGML_XDNA_TILE_N=64

# ── Slot 2: K=4096 ────────────────────────────────────────────────────────────
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
#        M=2048 K=4096 N=64 m=64 k=64 n=64 n_aie_cols=4
export GGML_XDNA_XCLBIN_PATH_2=~/xclbin-decode/k4096_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_2=~/xclbin-decode/k4096_n64_decode.txt
export GGML_XDNA_TILE_M2=2048
export GGML_XDNA_TILE_K2=4096
export GGML_XDNA_TILE_N2=64

# ── Slot 3: K=5632 (FFN down projections) ────────────────────────────────────
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
#        M=2048 K=5632 N=64 m=64 k=64 n=64 n_aie_cols=4
export GGML_XDNA_XCLBIN_PATH_3=~/xclbin-decode/k5632_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_3=~/xclbin-decode/k5632_n64_decode.txt
export GGML_XDNA_TILE_M3=2048
export GGML_XDNA_TILE_K3=5632
export GGML_XDNA_TILE_N3=64

# ── Slot 4: K=14336 (FFN gate/up projections) ────────────────────────────────
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
#        M=2048 K=14336 N=64 m=64 k=64 n=64 n_aie_cols=4
export GGML_XDNA_XCLBIN_PATH_4=~/xclbin-decode/k14336_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_4=~/xclbin-decode/k14336_n64_decode.txt
export GGML_XDNA_TILE_M4=2048
export GGML_XDNA_TILE_K4=14336
export GGML_XDNA_TILE_N4=64

# ── Tuning ────────────────────────────────────────────────────────────────────
# Per-tile kernel timeout in milliseconds (default: 5000).
# Increase if you see timeout errors on a thermally-throttled system.
# export GGML_XDNA_TIMEOUT_MS=5000
