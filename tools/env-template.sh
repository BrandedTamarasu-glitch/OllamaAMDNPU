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

# ── Build location ────────────────────────────────────────────────────────────
# Adjust REPO to wherever you cloned OllamaAMDNPU
REPO=~/OllamaAMDNPU

# ── Backend selection ─────────────────────────────────────────────────────────
# Points llama.cpp to the XDNA backend shared library.
# Change to libggml-vulkan.so to use the Vulkan iGPU backend instead.
export GGML_BACKEND_PATH=$REPO/build/bin/libggml-xdna.so

# ── Slot 1: K=2048 (attention Q/K/V/O projections) ───────────────────────────
# Required. Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p M=2048 K=2048 N=64 m=64 k=64 n=64 n_aie_cols=1
export GGML_XDNA_XCLBIN_PATH=~/xclbin/k2048.xclbin
export GGML_XDNA_INSTR_PATH=~/xclbin/k2048_sequence.bin
export GGML_XDNA_TILE_M=2048
export GGML_XDNA_TILE_K=2048
export GGML_XDNA_TILE_N=64

# ── Slot 2: K=4096 (mixed-width layers) ──────────────────────────────────────
# Optional. Adds coverage for layers with K=4096 inner dimension.
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p M=2048 K=4096 N=64 m=64 k=64 n=64 n_aie_cols=1
export GGML_XDNA_XCLBIN_PATH_2=~/xclbin/k4096.xclbin
export GGML_XDNA_INSTR_PATH_2=~/xclbin/k4096_sequence.bin
export GGML_XDNA_TILE_M2=2048
export GGML_XDNA_TILE_K2=4096
export GGML_XDNA_TILE_N2=64

# ── Slot 3: K=5632 (FFN down projections, LLaMA-3 8B) ───────────────────────
# Optional but recommended for LLaMA-3 8B — covers ~20% of compute.
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p M=2048 K=5632 N=64 m=64 k=64 n=64 n_aie_cols=1
export GGML_XDNA_XCLBIN_PATH_3=~/xclbin/k5632.xclbin
export GGML_XDNA_INSTR_PATH_3=~/xclbin/k5632_sequence.bin
export GGML_XDNA_TILE_M3=2048
export GGML_XDNA_TILE_K3=5632
export GGML_XDNA_TILE_N3=64

# ── Slot 4: K=14336 (FFN gate/up projections, LLaMA-3 8B) ───────────────────
# Optional but recommended for LLaMA-3 8B — covers ~45% of compute.
# Build command:
#   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p M=2048 K=14336 N=64 m=64 k=64 n=64 n_aie_cols=1
export GGML_XDNA_XCLBIN_PATH_4=~/xclbin/k14336.xclbin
export GGML_XDNA_INSTR_PATH_4=~/xclbin/k14336_sequence.bin
export GGML_XDNA_TILE_M4=2048
export GGML_XDNA_TILE_K4=14336
export GGML_XDNA_TILE_N4=64

# ── Tuning ────────────────────────────────────────────────────────────────────
# Minimum N dimension before bypassing the NPU (default: skip very small matmuls).
# Set to 2 to maximise NPU coverage (attempt everything >= N=2).
export GGML_XDNA_MIN_N=2

# Per-tile kernel timeout in milliseconds (default: 5000).
# Increase if you see timeout errors on a thermally-throttled system.
# export GGML_XDNA_TIMEOUT_MS=5000
