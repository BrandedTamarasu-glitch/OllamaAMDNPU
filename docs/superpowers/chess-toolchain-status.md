# Task 8: Chess Compiler Status

**Date:** 2026-03-31

## Summary

Chess is **not available** on this system. Peano (LLVM-AIE) is the sole active
AIE kernel compiler. All kernel builds in this project use Peano.

---

## Chess Compiler — NOT PRESENT

| Check | Result |
|-------|--------|
| `/opt/xilinx/xrt/amdxdna/` | **Does not exist** |
| `chess-cc` in PATH | **Not found** |
| Any `chess-cc` binary | Not found anywhere on system |

Chess is a proprietary Xilinx/AMD compiler delivered as part of Vitis. It is not
included in the open-source mlir-aie distribution and requires a separate Vitis
installation. This system has only the open-source toolchain installed.

The `chess_intrinsic_wrapper.{cpp,ll}` files in the mlir-aie tree are source
stubs; they compile via Peano when Chess is absent.

---

## Active Toolchain — Peano (LLVM-AIE)

| Component | Details |
|-----------|---------|
| Package | `mlir-aie 0.0.1.2026032304+6fc2408` |
| Location | `/home/corye/Claude/OllamaAMD/mlir-aie-env/lib/python3.14/site-packages/` |
| Peano clang | `llvm-aie/bin/clang` — clang 20.0.0 (Xilinx/llvm-aie `1f59472b`) |
| Host clang | `mlir-aie-env/bin/clang` — clang 21.1.8 (host x86) |
| Frontend | `aiecc` (Python), auto-discovers Peano via `VIRTUAL_ENV` |

### PEANO_INSTALL_DIR note

`env_setup.sh` sets `PEANO_INSTALL_DIR` to `mlir-aie/install/` which **does not
exist** on this system. Do not source `env_setup.sh`, or override immediately after:

```bash
export PEANO_INSTALL_DIR=/home/corye/Claude/OllamaAMD/mlir-aie-env/lib/python3.14/site-packages/llvm-aie
```

Alternatively, leave `PEANO_INSTALL_DIR` unset — `aiecc` discovers Peano via
`VIRTUAL_ENV` when the variable is absent.

### Verified working build

Single-core i8 matmul (64×2048×64, tile 64×64×64) compiles and runs:

```
Avg NPU matmul time: 125.5µs  (133.7 GOPS)
Min:  93µs  (180.4 GOPS)
Max: 340µs   (49.3 GOPS)
```

Xclbin: `programming_examples/basic/matrix_multiplication/single_core/build/final_64x2048x64_64x64x64.xclbin`
