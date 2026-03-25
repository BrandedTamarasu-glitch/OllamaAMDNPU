# llama.cpp — AMD XDNA2 NPU Backend (RyzenAI npu5)

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with a custom ggml backend that offloads matrix multiplication to the AMD XDNA2 NPU found in **Ryzen AI MAX** processors (e.g. Ryzen AI MAX 385).

The NPU backend accelerates `GGML_OP_MUL_MAT` (the dominant operation in LLM inference) via XRT kernel dispatch. All other operations fall back to the CPU backend automatically.

---

## Hardware

- **CPU/NPU**: AMD Ryzen AI MAX 385 (or similar Ryzen AI MAX / Strix Halo)
- **NPU**: RyzenAI-npu5 (XDNA2), exposed as `/dev/accel/accel0`
- **OS**: Linux (tested on CachyOS 6.19, kernel with amdxdna driver)
- **XRT**: 2.21.75+

---

## How It Works

The backend implements the [ggml backend interface](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h) with the following design:

- **`GGML_OP_MUL_MAT`** is dispatched to the NPU using a compiled `.xclbin` kernel built with [mlir-aie](https://github.com/Xilinx/mlir-aie).
- Weights are quantised to **int8** with per-row scales once on first use and cached — subsequent tokens reuse the cached result, eliminating re-quantisation overhead.
- Activations are quantised per-call (they change every token).
- The kernel operates on fixed-size tiles (`TILE_M × TILE_K × TILE_N`); larger matrices are tiled at the host level.
- A **second xclbin slot** can be loaded for layers with a different K dimension (e.g. K=5632 FFN down layers vs K=2048 attention layers).

---

## Prerequisites

1. **amdxdna kernel driver** — loaded and `/dev/accel/accel0` accessible
2. **XRT** (Xilinx Runtime) — headers and shared libraries installed
3. **Compiled xclbins** — built with mlir-aie (see below)
4. **Unlimited memlock** — required for XRT DMA buffer pinning

### Setting memlock to unlimited

```bash
# Create the systemd override (one-time, requires sudo)
sudo mkdir -p /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/memlock.conf <<EOF
[Service]
LimitMEMLOCK=infinity
EOF
```

Log out and back in. Verify with:
```bash
ulimit -Hl   # must show: unlimited
```

---

## Building the xclbins

The xclbins are built with mlir-aie's `matrix_multiplication` example. For K=2048 (attention layers) and K=5632 (FFN down layers) with 1-core dispatch:

```bash
cd mlir-aie/programming_examples/basic/matrix_multiplication/single_core

# K=2048 (attention)
make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
     M=2048 K=2048 N=64 m=64 k=64 n=64 n_aie_cols=1

# K=5632 (FFN down)
make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
     M=2048 K=5632 N=64 m=64 k=64 n=64 n_aie_cols=1
```

The build produces a `.xclbin` and `_sequence.bin` (instruction file) for each configuration.

---

## Building llama.cpp with the XDNA backend

```bash
cmake -B build \
  -DGGML_XDNA=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build build --parallel
```

---

## Running

### Environment variables

**Slot 1 (required):**
```bash
export GGML_XDNA_XCLBIN_PATH=/path/to/k2048.xclbin
export GGML_XDNA_INSTR_PATH=/path/to/k2048_sequence.bin
export GGML_XDNA_TILE_M=2048
export GGML_XDNA_TILE_K=2048
export GGML_XDNA_TILE_N=64
```

**Slot 2 (optional — adds K=5632 FFN layer coverage):**
```bash
export GGML_XDNA_XCLBIN_PATH_2=/path/to/k5632.xclbin
export GGML_XDNA_INSTR_PATH_2=/path/to/k5632_sequence.bin
export GGML_XDNA_TILE_K2=5632
export GGML_XDNA_TILE_M2=2048
export GGML_XDNA_TILE_N2=64
```

**Optional tuning:**
```bash
export GGML_XDNA_MIN_BATCH=32768   # minimum M×N×K to offload (default 32768)
export GGML_XDNA_TIMEOUT_MS=5000   # per-tile NPU timeout in ms (default 5000)
```

### Running inference

```bash
./build/bin/llama-cli \
  -m models/your-model.gguf \
  -n 512 \
  -p "Your prompt here"
```

On successful init you should see:
```
ggml_xdna: kernel ready — tile 2048×2048×64
ggml_xdna: slot 2 ready — tile 2048×5632×64
```

---

## All supported environment variables

| Variable | Default | Description |
|---|---|---|
| `GGML_XDNA_XCLBIN_PATH` | — | Path to slot 1 `.xclbin` (required) |
| `GGML_XDNA_INSTR_PATH` | — | Path to slot 1 `_sequence.bin` (required) |
| `GGML_XDNA_TILE_M` | 32 | Slot 1 tile rows |
| `GGML_XDNA_TILE_K` | 32 | Slot 1 tile inner dimension |
| `GGML_XDNA_TILE_N` | 32 | Slot 1 tile cols |
| `GGML_XDNA_MIN_BATCH` | 32768 | Min M×N×K to offload to NPU |
| `GGML_XDNA_TIMEOUT_MS` | 5000 | Per-tile kernel timeout (ms) |
| `GGML_XDNA_XCLBIN_PATH_2` | — | Path to slot 2 `.xclbin` (optional) |
| `GGML_XDNA_INSTR_PATH_2` | — | Path to slot 2 `_sequence.bin` (optional) |
| `GGML_XDNA_TILE_K2` | — | Slot 2 K dimension (required to enable slot 2) |
| `GGML_XDNA_TILE_M2` | 32 | Slot 2 tile rows |
| `GGML_XDNA_TILE_N2` | 32 | Slot 2 tile cols |

---

## Performance (Ryzen AI MAX 385, Meta-Llama-3.1-8B-Instruct Q4_K_M)

### Prefill throughput vs context length

All measurements: Meta-Llama-3.1-8B-Instruct Q4_K_M, `llama-bench -r 1 --no-warmup`, Ryzen AI MAX 385.

| Backend | pp=512 | pp=2048 | pp=4096 | pp=8192 | Decode |
|---------|--------|---------|---------|---------|--------|
| CPU only | 4.5 t/s | 4.4 t/s | 4.0 t/s | 3.6 t/s | ~4.4 t/s |
| NPU (4 slots, ub=512) | 10.2 t/s | **12.9 t/s** | 11.7 t/s | 8.9 t/s | ~4.1 t/s |
| Vulkan iGPU (KHR_coopmat, ngl=99) | 895 t/s | 849 t/s | 776 t/s | 657 t/s | ~43 t/s |

Notes:
- **NPU is 2–3× faster than CPU** at all context lengths, with peak efficiency at pp=2048 (one full XDNA2 tile)
- NPU prefill degrades at long context because attention score matmuls (K=seq_len, variable) fall back to CPU; only fixed-K projection matmuls offload to NPU
- `--ubatch-size 2048` does **not** improve NPU throughput — larger CPU-side attention batches (O(n²)) outweigh better tile utilisation
- Vulkan degrades ~27% from 512→8192 tokens; NPU degrades ~13% (lower absolute speed but more context-resilient)
- 8k context validated without OOM (KV cache ≈ 1 GiB; model ≈ 4.6 GiB; 30 GiB RAM is sufficient)

### Power (SoC PPT, measured via `tools/bench-power.sh`)

| Backend | Avg power | J/token (decode) | Notes |
|---------|-----------|------------------|-------|
| NPU | 45.8 W | 11.2 J/tok | Dedicated XDNA2 silicon — does not contend with iGPU |
| Vulkan | 67.9 W | 1.6 J/tok | 7× better efficiency/token due to much higher decode speed |

### When to use NPU vs Vulkan

**Use Vulkan** when the GPU is idle and you want maximum throughput (40–200× faster prefill).

**Use NPU** when:
- The iGPU is busy (gaming, rendering, GPU compute) — NPU runs on dedicated XDNA2 silicon with no GPU contention
- Thermal/noise matters — NPU draws ~22W less SoC power, keeping fans quieter during long sessions
- Running inference in the background alongside GPU workloads

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | NPU baseline — XDNA2 backend, weight cache, K=2048 (TinyLlama 1.1B) |
| 2 | ✅ Done | Dual-slot dispatch — second xclbin slot for K=5632 (FFN down layers) |
| 3 | ✅ Done | 8B model support — K=4096 and K=14336 slots, tile-loop optimisation (+15–28% prefill over CPU) |
| 4 | ✅ Done | Workload-isolation characterisation — power measurement vs Vulkan; NPU confirmed as non-competing backend (dedicated XDNA2 silicon); `bench-power.sh` tooling |
| 5 | ✅ Done | Long context (8k–32k) — validated at 8k with 30 GiB RAM; NPU 2–3× over CPU; peaks at pp=2048; attention falls to CPU at long context (variable-K not covered by xclbins); `bench-context.sh` tooling |

---

## AI Assistance

This project was designed and implemented with the assistance of **Claude Sonnet 4.6** (`claude-sonnet-4-6`) by [Anthropic](https://www.anthropic.com), accessed via [Claude Code](https://claude.ai/code).

AI assistance covered: backend architecture, XRT kernel dispatch design, tile loop optimisation, performance debugging, benchmarking tooling, and documentation.

---

## Upstream

This repo tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The XDNA backend lives entirely in `ggml/src/ggml-xdna/`.
