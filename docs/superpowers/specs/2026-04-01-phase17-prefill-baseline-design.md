# Phase 17: NPU Prefill Baseline Measurement

**Date:** 2026-04-01
**Status:** Approved for implementation
**Scope:** Measure-only — no kernel builds, no new tooling

## Goal

Determine whether NPU prefill is worth investing in before committing to
dedicated prefill xclbin builds. A single benchmark run with existing
decode xclbins (tile_n=64, mm.cc-based) establishes a lower-bound NPU
prefill t/s at pp=160 to compare against Vulkan's 16.41 t/s baseline.

## Background

The XDNA backend supports a two-tier slot system (prefill + decode) but
no prefill xclbins exist yet. The existing decode xclbins (~/xclbin-decode/,
tile_n=64) handle N>1 ops at ~17% padding waste for N=160 (3 tile columns).
GGML_XDNA_MAX_N=1 currently blocks all prefill from the NPU; removing it
enables measurement without any build cycle.

## Setup

```bash
source ~/.npu-decode.env          # xclbin paths, tile dims (TILE_N=64)
unset GGML_XDNA_MAX_N             # remove decode-only cap
export GGML_XDNA_MIN_N=2          # N=1 decode → CPU; N=160 prefill → NPU
export GGML_VK_VISIBLE_DEVICES="" # disable Vulkan — NPU + CPU only
```

## Measurement

```bash
MODEL=~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf \
  ./build/bin/llama-bench -m "$MODEL" -p 160 -n 1 -r 3 2>/dev/null
```

Three repetitions for variance. The `pp 160` row gives NPU prefill t/s.
The `tg 1` row is CPU decode — not relevant to this measurement.

## Decision Gate

| NPU prefill t/s | Interpretation | Next step |
|---|---|---|
| ≥ 8 t/s (≥ 0.5× Vulkan 16.41) | Competitive at tile_n=64 | Phase 17B: build tile_n=128/256 prefill xclbins |
| 2–8 t/s | Sub-optimal tiling — tile_n=64 is wasteful for N=160 | bench-prefill.sh sweep across tile_n and pp values |
| < 2 t/s | Dispatch latency dominates even at N=160 | Short characterization doc; pause NPU prefill work |

## Output

Results recorded in `docs/phase-17-prefill-baseline.md` (new file).
No changes to tools, kernels, or xclbins in this phase.

## Out of Scope

- Building new prefill xclbins
- Modifying mm.cc or any AIE kernel
- Power measurement (covered by Phase 16 methodology if needed later)
- Sweep across multiple pp values or tile sizes
