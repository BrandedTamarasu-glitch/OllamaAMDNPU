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

## Performance (Ryzen AI MAX 385, TinyLlama 1.1B Q4_K_M)

| Backend | Prompt t/s | Generation t/s |
|---|---|---|
| CPU baseline | ~31 | ~31 |
| NPU (K=2048 only, with weight cache) | ~33 | ~28 |

> Note: TinyLlama 1.1B is used for development testing only — it is too small for reliable output quality. Larger models (7B+) will show more significant NPU coverage across layers.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | NPU baseline — XDNA2 backend, weight cache, K=2048 (TinyLlama 1.1B) |
| 2 | ✅ Done | Dual-slot dispatch — second xclbin slot for K=5632 (FFN down layers) |
| 3 | ✅ Done | 8B model support — K=4096 and K=14336 slots, tile-loop optimisation (+15–28% prefill over CPU) |
| 4 | Planned | Workload-isolation mode — NPU as non-competing backend (doesn't contend with iGPU); power measurement vs Vulkan; GGML_XDNA_MAX_N for explicit NPU-only or Vulkan-only routing |
| 5 | Planned | Long context (8k–32k) — validate KV cache memory, RoPE scaling, benchmark NPU prefill at full tile utilisation (N≈2048) |

---

## Upstream

This repo tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The XDNA backend lives entirely in `ggml/src/ggml-xdna/`.
