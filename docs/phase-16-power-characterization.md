# Phase 16: Power Characterization

**Date:** 2026-04-01
**Model:** Meta-Llama-3-8B-Instruct Q8_0
**Hardware:** Ryzen AI MAX 385 · XDNA2 NPU
**Methodology:** bench-power.sh with 5s warmup exclusion, 5s idle baseline, 50ms sampling
**Source data:** `docs/phase-16b-final.json` (NPU), `docs/phase-16b-vulkan.json` (Vulkan)

## Results

| Backend | Mode | t/s | Avg Power (W) | Idle (W) | Marginal (W) | J/tok (total) | J/tok (marginal) |
|---------|------|-----|---------------|----------|--------------|---------------|------------------|
| NPU (Phase 14/15, v1 kernel) | decode | 0.12 | 41.16 | 34.24 | 6.92 | 343.00 | 57.67 |
| Vulkan (iGPU, -ngl 99) | decode | 6.40 | 96.94 | 36.01 | 60.93 | 15.15 | 9.52 |

Prefill (Vulkan only, pp=160): 16.41 t/s. NPU runs decode-only (p=0, ub=1); no prefill measurement.

### Tile Configuration (NPU)

| Param | Value |
|-------|-------|
| TILE_M | 128 |
| TILE_K | 2048 (slot 1) / 4096 / 5632 / 14336 |
| TILE_N | 64 |
| xclbin hash | 32f81da27ba8 |

## Analysis

At current NPU throughput (0.12 t/s), energy efficiency is far below Vulkan.

**Total J/tok comparison:**
- NPU: 343.0 J/tok
- Vulkan: 15.1 J/tok
- Vulkan is **22.7× more efficient** (total)

**Marginal J/tok comparison** (above SoC idle):
- NPU: 57.7 J/tok
- Vulkan: 9.5 J/tok
- Vulkan is **6.1× more efficient** (marginal)

The NPU would need approximately **22.7× throughput improvement at constant power**
to match Vulkan on total J/tok, or **6.1×** to match on marginal J/tok.

Note: Vulkan draws 60.93W above idle vs NPU 6.92W above idle. The NPU is far more
power-constrained but also far slower — the throughput deficit dominates.

## Notes on Phase 12 Comparison

No clean Phase 12 power baseline exists for Q8_0. The 1.40 t/s figure referenced in
planning documents was a measurement artifact: the shell environment had XDNA still
dispatching N=1 ops, inflating apparent "CPU" throughput and contaminating the NPU
measurement. Correct NPU decode rate established in Phase 14B: **0.12 t/s steady-state**.

The Phase 9 measurement (0.65 t/s, Q4_K_M) is not directly comparable — Q8_0 has more
NPU-eligible GEMMs than Q4_K_M, driving dispatch count up and t/s down further.

## Value of This Data

1. **Baseline for future kernel work:** Any NPU kernel improvements can be benchmarked
   against 0.12 t/s / 343 J/tok / 57.7 J/tok(marginal) with identical methodology.
2. **Marginal NPU power isolated:** 6.92W above SoC idle establishes that the NPU itself
   draws modestly; the efficiency problem is dispatch latency, not power budget.
3. **Vulkan reference point:** 15.15 J/tok (total) / 9.52 J/tok (marginal) is the
   target for NPU to become competitive on energy efficiency.
4. **Measurement methodology locked:** bench-power.sh with warmup exclusion, idle
   subtraction, and 50ms sampling is now the standard for all future power work.

## Metadata Notes

The `tile_config` and `xclbin_hash` fields in `phase-16b-vulkan.json` reflect residual
XDNA env vars from the shell session, not Vulkan configuration. These fields are N/A
for Vulkan runs; JSON consumers should ignore them when `backend == "--vulkan"`.
