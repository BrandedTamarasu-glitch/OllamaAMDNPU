# Phase 17B Results — NPU Prefill tile_n=128

**Date:** 2026-04-01  
**Status:** GATE PASSED — Phase 17C authorized

---

## Result

| Config | Backend | pp160 t/s |
|---|---|---|
| Phase 17 baseline — Vulkan alone | Vulkan | 16.41 |
| Phase 17 baseline — Vulkan+XDNA tile_n=64 | Vulkan+XDNA | 6.11 |
| **Phase 17B — XDNA tile_n=128** | **XDNA** | **17.70 ± 1.98** |

NPU-only prefill at tile_n=128 beats Vulkan alone by **+7.8%**.

> ±1.98 is the std dev reported by llama-bench across 3 runs (CV ≈ 11.2% — high but expected at pp=160
> due to NPU dispatch jitter at small token counts). Phase 17C sweep will use -r 5 for tighter estimates.

---

## Gate Decision

| t/s threshold | Decision |
|---|---|
| ≥ 8 | Phase 17C: sweep pp values + tile_n=256 |
| 5–8 | One tile_n=256 build then re-evaluate |
| 2–5 | Close — dispatch latency architectural |
| < 2 | Close — fundamental floor confirmed |

**Outcome: 17.70 t/s → Phase 17C full sweep authorized**

---

## Configuration

- Model: `~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf` (8B Q8_0)
- Bench: `llama-bench -p 160 -n 0 -r 3`
- Backend: XDNA only (Vulkan SO physically hidden during measurement)
- Env: `~/.npu-prefill.env` (8-slot: prefill slots 1-4, decode slots 5-8)

### Xclbin config (slots 1-4, prefill)
- TILE_M=128, TILE_N=128
- Inner tiles: m=32 k=128 n=32 (constrained: M=128/4 AIE rows, N=128/4 AIE cols)
- Device: npu2 (aie.device(npu2))
- mlir-aie commit: `7721c20`
- K values: 2048 / 4096 / 5632 / 14336
- Built with: `make all M=128 K=$K N=128 m=32 k=128 n=32 NPU2=1`

---

## Reproduction

```bash
# 1. Build xclbins (if not already present)
bash tools/build-prefill-xclbins.sh

# 2. Validate env
bash tools/validate-prefill-env.sh

# 3. Measure (Vulkan hidden, prefill only)
export GGML_VK_VISIBLE_DEVICES=""
source ~/.npu-prefill.env
# Restore SO on exit/interrupt — without this, subsequent runs would appear NPU-only when they aren't
trap 'mv build/bin/libggml-vulkan.so.bench-hidden build/bin/libggml-vulkan.so 2>/dev/null; exit' ERR EXIT
mv build/bin/libggml-vulkan.so build/bin/libggml-vulkan.so.bench-hidden
./build/bin/llama-bench -m ~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -p 160 -n 0 -r 3
mv build/bin/libggml-vulkan.so.bench-hidden build/bin/libggml-vulkan.so
trap - ERR EXIT
```

---

## Notes

- Phase 17 baseline (tile_n=64) was net-negative (+Vulkan) due to dispatch overhead exceeding XDNA compute advantage
- tile_n=128 eliminates the overhead gap: NPU alone now slightly faster than Vulkan alone
- Inner tile constraint: with M=128, n_aie_rows=4 forces max m=32; n=32 from N=128/4 cols
- m=128 inner tiles (as used in decode) are not valid for M=128 prefill — `M % (m * n_aie_rows)` fails
- Insts file size: 1632 bytes (vs 5168 bytes for tile_n=64 decode) — 4-byte aligned ✓
- xclbin sizes vary by K (96K–144K range) — K=4096 largest at 143K
- Decode phase (N=1) not measured: with Vulkan hidden and no tile_n=64 selected by find_slot fallback, decode would use tile_n=128 with zero-padding; excluded via `-n 0`

---

## Next — Phase 17C

Sweep targets:
1. Different pp values (pp32, pp64, pp128, pp256, pp512) to characterize prefill throughput curve
2. tile_n=256 build (if M constraint allows — requires N=256 with 4 cols → n=64 inner)
3. Compare NPU prefill t/s vs Vulkan across the pp curve
4. Target: establish minimum pp where NPU prefill is ≥ Vulkan
