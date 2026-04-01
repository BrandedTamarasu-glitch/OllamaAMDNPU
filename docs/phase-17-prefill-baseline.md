# Phase 17: NPU Prefill Baseline

**Date:** 2026-04-01
**Model:** Meta-Llama-3-8B-Instruct Q8_0
**Hardware:** Ryzen AI MAX 385 · XDNA2 NPU
**Methodology:** Existing decode xclbins (tile_n=64, mm.cc), MIN_N=2, Vulkan disable attempted

## Setup

- xclbin: ~/xclbin-decode/ (k2048/k4096/k5632/k14336, tile_n=64)
- GGML_XDNA_MIN_N=2 (N=1 decode → non-NPU; N=160 prefill → NPU eligible)
- GGML_XDNA_MAX_N: unset (no upper cap)
- GGML_VK_VISIBLE_DEVICES="" (attempted Vulkan disable — ineffective, see note)

## Results

| Test | t/s (mean ± stddev) | reps |
|------|---------------------|------|
| Prefill pp=160 (Vulkan+XDNA) | 6.11 ± 0.17 | 3 |
| Decode tg=1 (Vulkan, ngl=99 spill) | 0.22 ± 0.00 | 3 |

**Vulkan-alone prefill baseline (Phase 16):** 16.41 t/s
**Combined (Vulkan+XDNA) / Vulkan-alone ratio:** 0.37× — XDNA is slowing prefill by 63%

## Measurement Caveat

`GGML_VK_VISIBLE_DEVICES=""` does **not** disable Vulkan. The llama-bench output
confirmed `backend: Vulkan,XDNA` — both were active. To get a clean NPU-only prefill
number, the Vulkan SO must be physically hidden (as bench-power.sh `--npu` mode does
via mv libggml-vulkan.so → .bench-hidden).

This means the 6.11 t/s is a *combined* Vulkan+XDNA measurement. The NPU was claiming
N=160 GEMM ops (3 tile columns at tile_n=64) while Vulkan also ran. The combined
throughput being 63% below Vulkan-alone confirms XDNA dispatch overhead at tile_n=64
is a net negative for prefill.

The decode tg=1 value (0.22 t/s) is anomalously slow — Vulkan with ngl=99 on an iGPU
attempting to hold a Q8_0 8B model (7.95 GiB) appears to spill to system memory.

## Decision Gate Outcome

**Gate hit: 2–8 t/s range (combined), effectively < 2 t/s for NPU contribution alone**

The combined measurement is slower than Vulkan alone. XDNA at tile_n=64 is a
net negative for prefill at pp=160. Two paths forward:

1. **Build larger tile_n xclbins (tile_n=128 or 256):** Reduce dispatch count from
   3 calls/GEMM to 1–2 calls/GEMM at N=160. Get a clean NPU-only number by hiding
   the Vulkan SO before measuring.

2. **Accept Vulkan for prefill:** NPU decode is retired (Phase 14). NPU prefill at
   tile_n=64 is counterproductive. Declare Vulkan as the permanent prefill backend
   and close NPU prefill investigation.

## Notes

- tile_n=64 produces 3 dispatch calls per GEMM at N=160 (~17% padding waste + 3× dispatch overhead)
- A tile_n=128 xclbin would reduce to 2 calls; tile_n=256 to 1 call (with 37.5% padding at N=160)
- Clean NPU-only measurement requires: `mv build/bin/libggml-vulkan.so .bench-hidden` before benchmarking
- This measurement is a lower bound on combined Vulkan+XDNA throughput; pure NPU would be worse
