# Phase 17B: NPU Prefill Xclbin Build (tile_n=128)

**Date:** 2026-04-01
**Status:** Approved for implementation
**Scope:** Build 4 prefill xclbins, configure two-tier env, run clean NPU-only prefill measurement

## Goal

Build tile_n=128 prefill xclbins and get a clean NPU-only prefill t/s at pp=160
(Vulkan SO hidden). Determine whether NPU prefill is viable before further investment.

## Background

Phase 17 revealed that `GGML_VK_VISIBLE_DEVICES=""` does not disable Vulkan — the SO
must be physically hidden. The combined Vulkan+XDNA result (6.11 t/s) was 63% slower
than Vulkan-alone (16.41 t/s) at tile_n=64 (3 dispatch calls per GEMM at N=160).

tile_n=128 reduces dispatches per GEMM from 3 to 2 at N=160 (33% reduction). The
XDNA backend already supports 8 slots in a two-tier layout: slots 1–4 for prefill,
slots 5–8 for decode. The existing decode xclbins (~/xclbin-decode/, tile_n=64) stay
in slots 5–8 unchanged.

## Build

**Script:** `tools/build-prefill-xclbins.sh`

Wraps 4 invocations of `whole_array.py` (same pipeline as `build-trace-xclbin.sh`):

| Slot | K     | TILE_M | TILE_N | Output file |
|------|-------|--------|--------|-------------|
| 1    | 2048  | 128    | 128    | ~/xclbin-prefill/k2048_n128_prefill.xclbin |
| 2    | 4096  | 128    | 128    | ~/xclbin-prefill/k4096_n128_prefill.xclbin |
| 3    | 5632  | 128    | 128    | ~/xclbin-prefill/k5632_n128_prefill.xclbin |
| 4    | 14336 | 128    | 128    | ~/xclbin-prefill/k14336_n128_prefill.xclbin |

Each build also produces an `_insts.txt` and a JSON manifest (matching trace script pattern).

## Environment Configuration

Two-tier env file: `~/.npu-prefill.env`

```bash
# Prefill slots 1-4 (tile_n=128)
export GGML_XDNA_XCLBIN_PATH=~/xclbin-prefill/k2048_n128_prefill.xclbin
export GGML_XDNA_INSTR_PATH=~/xclbin-prefill/k2048_n128_prefill_insts.txt
export GGML_XDNA_TILE_M=128
export GGML_XDNA_TILE_K=2048
export GGML_XDNA_TILE_N=128

export GGML_XDNA_XCLBIN_PATH_2=~/xclbin-prefill/k4096_n128_prefill.xclbin
export GGML_XDNA_INSTR_PATH_2=~/xclbin-prefill/k4096_n128_prefill_insts.txt
export GGML_XDNA_TILE_M2=128; export GGML_XDNA_TILE_K2=4096; export GGML_XDNA_TILE_N2=128

export GGML_XDNA_XCLBIN_PATH_3=~/xclbin-prefill/k5632_n128_prefill.xclbin
export GGML_XDNA_INSTR_PATH_3=~/xclbin-prefill/k5632_n128_prefill_insts.txt
export GGML_XDNA_TILE_M3=128; export GGML_XDNA_TILE_K3=5632; export GGML_XDNA_TILE_N3=128

export GGML_XDNA_XCLBIN_PATH_4=~/xclbin-prefill/k14336_n128_prefill.xclbin
export GGML_XDNA_INSTR_PATH_4=~/xclbin-prefill/k14336_n128_prefill_insts.txt
export GGML_XDNA_TILE_M4=128; export GGML_XDNA_TILE_K4=14336; export GGML_XDNA_TILE_N4=128

# Decode slots 5-8 (tile_n=64, existing)
export GGML_XDNA_XCLBIN_PATH_5=~/xclbin-decode/k2048_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_5=~/xclbin-decode/k2048_n64_decode.txt
export GGML_XDNA_TILE_M5=128; export GGML_XDNA_TILE_K5=2048; export GGML_XDNA_TILE_N5=64

export GGML_XDNA_XCLBIN_PATH_6=~/xclbin-decode/k4096_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_6=~/xclbin-decode/k4096_n64_decode.txt
export GGML_XDNA_TILE_M6=128; export GGML_XDNA_TILE_K6=4096; export GGML_XDNA_TILE_N6=64

export GGML_XDNA_XCLBIN_PATH_7=~/xclbin-decode/k5632_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_7=~/xclbin-decode/k5632_n64_decode.txt
export GGML_XDNA_TILE_M7=128; export GGML_XDNA_TILE_K7=5632; export GGML_XDNA_TILE_N7=64

export GGML_XDNA_XCLBIN_PATH_8=~/xclbin-decode/k14336_n64_decode.xclbin
export GGML_XDNA_INSTR_PATH_8=~/xclbin-decode/k14336_n64_decode.txt
export GGML_XDNA_TILE_M8=128; export GGML_XDNA_TILE_K8=14336; export GGML_XDNA_TILE_N8=64

# Dispatch range: prefill (N≥128) → slots 1-4; decode (N<128) → slots 5-8
export GGML_XDNA_MIN_N=1
unset GGML_XDNA_MAX_N
```

`find_slot()` selects the slot with the largest tile_n ≤ N, so:
- N=160 → tile_n=128 slot (prefill) ✓
- N=1 → tile_n=64 slot (decode, smallest > N fallback) ✓

## Measurement

```bash
source ~/.npu-prefill.env
mv build/bin/libggml-vulkan.so build/bin/libggml-vulkan.so.bench-hidden
MODEL=~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf
./build/bin/llama-bench -m "$MODEL" -p 160 -n 1 -r 3 2>/dev/null
mv build/bin/libggml-vulkan.so.bench-hidden build/bin/libggml-vulkan.so
```

Confirm backend column shows `XDNA` only (not `Vulkan,XDNA`).

## Decision Gate

| NPU-only prefill t/s | Interpretation | Next step |
|---|---|---|
| ≥ 8 t/s (≥ 0.5× Vulkan) | Competitive — worth pursuing | Phase 17C: sweep pp values, build tile_n=256 |
| 2–8 t/s | Reduced overhead vs tile_n=64 but not competitive | Document, consider tile_n=256 |
| < 2 t/s | Dispatch latency is fundamental, not tile-size-specific | Close NPU prefill investigation |

## Output

Results recorded in `docs/phase-17b-prefill-xclbin.md`.

## Out of Scope

- Modifying mm.cc or any AIE kernel source
- Power measurement (Phase 16 methodology available if needed later)
- Sweeping multiple pp values (Phase 17C if gate passes)
- Two-tier live inference (prefill+decode combined pipeline)
