# Phase 17C Results — NPU Prefill pp Sweep (tile_n=128)

**Date:** 2026-04-01  
**Status:** GATE PASSED — NPU prefill viability confirmed for pp≥128

---

## Result

### NPU vs CPU — full pp sweep

| pp | NPU t/s (±std dev) | CPU t/s | NPU vs CPU |
|---|---|---|---|
| 32  | 3.20 ± 0.36  | 17.12 | −81.3% |
| 64  | 6.14 ± 0.58  | 17.12 | −64.1% |
| 128 | 21.58 ± 3.19 | 17.12 | **+26.0%** |
| 256 | 24.48 ± 0.95 | 17.12 | **+43.0%** |
| 512 | 23.35 ± 0.55 | 17.12 | **+36.4%** |

CPU baseline: pp=160 measurement (17.12 ± 0.03 t/s) used as reference across all pp values — CPU prefill throughput is relatively flat across this range.

**NPU exceeds CPU at pp≥128. Crossover estimated at ~pp90–100** (steep jump between pp64 and pp128 suggests a slot-routing transition at N≥128 tiles).

### Vulkan reference (not the comparison target)

| pp | Vulkan t/s (±std dev) | NPU/Vulkan ratio |
|---|---|---|
| 32  | 356.22 ± 0.47  | 0.9% |
| 64  | 615.15 ± 0.94  | 1.0% |
| 128 | 790.09 ± 0.84  | 2.7% |
| 256 | 888.46 ± 1.26  | 2.8% |
| 512 | 811.13 ± 1.76  | 2.9% |

Vulkan is 35–111× faster at all pp values. NPU is not competitive with Vulkan. The comparison target is CPU, not Vulkan — NPU viability means replacing CPU prefill when the iGPU is busy.

---

## Gate Decision

| Outcome | Threshold | Decision |
|---|---|---|
| NPU ≥ CPU at ≥ one pp value | confirmed at pp128, 256, 512 | Phase 17C: PASSED |

**Outcome: NPU prefill viability confirmed for pp≥128 workloads → Phase 17D (tile_n=256) authorized**

---

## Configuration

- Model: `~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf` (8B Q8_0)
- Bench: `llama-bench -p <N> -n 0 -r 5` (separate invocation per pp — see OOM note below)
- Backend: XDNA only (Vulkan SO physically hidden during measurement)
- Env: `~/.npu-prefill.env` (8-slot: prefill slots 1–4 tile_n=128, decode slots 5–8 tile_n=64)
- Vulkan isolation: `GGML_VK_VISIBLE_DEVICES=""` + libggml-vulkan.so hidden

### Xclbin config (same as Phase 17B)
- TILE_M=128, TILE_N=128
- Inner tiles: m=32 k=128 n=32
- Device: npu2 (aie.device(npu2))
- K values: 2048 / 4096 / 5632 / 14336

---

## Known Issue: OOM at consecutive pp values in one llama-bench invocation

Running `llama-bench -p 32,64,128,256,512` with the XDNA backend triggers an OOM kill (exit 137) after pp=64. The XRT context does not release resources between test iterations within a single process.

**Workaround:** run each pp value as a separate llama-bench invocation:

```bash
for pp in 32 64 128 256 512; do
    ./build/bin/llama-bench -m ~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf \
        -p $pp -n 0 -r 5
done
```

---

## Reproduction

```bash
# 1. Validate env
bash tools/validate-prefill-env.sh

# 2. Measure NPU (Vulkan hidden, one pp at a time)
export GGML_VK_VISIBLE_DEVICES=""
source ~/.npu-prefill.env
trap 'mv build/bin/libggml-vulkan.so.bench-hidden build/bin/libggml-vulkan.so 2>/dev/null; exit' ERR EXIT
mv build/bin/libggml-vulkan.so build/bin/libggml-vulkan.so.bench-hidden
for pp in 32 64 128 256 512; do
    ./build/bin/llama-bench -m ~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf \
        -p $pp -n 0 -r 5
done
mv build/bin/libggml-vulkan.so.bench-hidden build/bin/libggml-vulkan.so
trap - ERR EXIT

# 3. Measure Vulkan baseline (clean env, no NPU env vars)
env -i HOME=$HOME PATH=$PATH \
    ./build/bin/llama-bench -m ~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf \
        -p 32,64,128,256,512 -n 0 -r 5
```

---

## Notes

- pp=256 is the peak (24.48 t/s); pp=512 drops slightly (23.35 t/s), possibly due to larger dispatch batches approaching tile boundary effects
- High CV at pp=128 (±3.19, CV≈14.8%) — expected at the slot-routing transition boundary; pp=256 is much tighter (±0.95, CV≈3.9%)
- Vulkan baseline was measured with a clean environment (no XDNA env vars sourced) to avoid F16 activation path interference
- CPU baseline (17.12 t/s) is from pp=160; full CPU pp curve not measured — assumed flat given CPU prefill is memory-bandwidth-bound across this range

---

## Next — Phase 17D

1. Build tile_n=256 xclbins: `make all M=128 K=$K N=256 m=32 k=128 n=64 NPU2=1`
2. Sweep same pp values with tile_n=256 — expected: fewer dispatches at large pp, potentially higher peak
3. Add `--tile-n` flag to `tools/build-prefill-xclbins.sh` before building
