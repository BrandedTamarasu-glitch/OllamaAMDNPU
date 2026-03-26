#!/usr/bin/env python3
"""
int4 quantization sanity check.

For weight tensors from the Q4_K_M model:
  1. Dequantize to float32 (ground truth)
  2. Requantize to int8 symmetric per-block (current NPU path)
  3. Requantize to int4 symmetric per-block (proposed NPU path)
  4. Measure and compare reconstruction error
"""

import sys
import numpy as np
import gguf
from gguf import GGUFReader, GGUFValueType

MODEL = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
BLOCK_SIZE = 32   # quantization block size (same as current backend)
# Tensors to sample (mix of layer types)
TARGET_NAMES = [
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",
    "blk.15.attn_q.weight",
    "blk.15.ffn_gate.weight",
    "blk.31.attn_q.weight",
    "blk.31.ffn_down.weight",
]


def quant_int8_symmetric(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-block symmetric int8 quantization. Returns (dequantized, scales)."""
    assert x.ndim == 1
    x = x.astype(np.float32)
    n = len(x)
    pad = (-n) % BLOCK_SIZE
    xp = np.pad(x, (0, pad))
    blocks = xp.reshape(-1, BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = amax / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(blocks / scales), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scales).reshape(-1)[:n]
    return deq, scales.flatten()


def quant_int4_symmetric(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-block symmetric int4 quantization (range -8..7). Returns (dequantized, scales)."""
    assert x.ndim == 1
    x = x.astype(np.float32)
    n = len(x)
    pad = (-n) % BLOCK_SIZE
    xp = np.pad(x, (0, pad))
    blocks = xp.reshape(-1, BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = amax / 7.0      # int4 signed max = 7
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)
    deq = (q.astype(np.float32) * scales).reshape(-1)[:n]
    return deq, scales.flatten()


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal = np.mean(original ** 2)
    noise = np.mean((original - reconstructed) ** 2)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def max_abs_err(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.max(np.abs(original - reconstructed)))


def main():
    print(f"Loading {MODEL} ...")
    reader = GGUFReader(MODEL)

    tensors_by_name = {t.name: t for t in reader.tensors}

    print(f"\n{'Tensor':<40} {'dtype':<12} {'shape':<20} {'int8 RMSE':>10} {'int4 RMSE':>10} {'int8 SNR':>9} {'int4 SNR':>9} {'ratio':>7}")
    print("-" * 130)

    all_int8_rmse = []
    all_int4_rmse = []
    all_int8_snr  = []
    all_int4_snr  = []

    for name in TARGET_NAMES:
        if name not in tensors_by_name:
            print(f"  {name}: NOT FOUND, skipping")
            continue

        t = tensors_by_name[name]
        dtype_name = t.tensor_type.name
        shape = tuple(t.shape)

        # Dequantize to float32 via gguf
        raw = np.array(t.data)
        # gguf reader already dequantizes to numpy float32 for Q4_K_M
        # Flatten to 1D
        f32 = raw.astype(np.float32).flatten()

        i8_deq, _ = quant_int8_symmetric(f32)
        i4_deq, _ = quant_int4_symmetric(f32)

        r8  = rmse(f32, i8_deq)
        r4  = rmse(f32, i4_deq)
        s8  = snr_db(f32, i8_deq)
        s4  = snr_db(f32, i4_deq)
        ratio = r4 / r8 if r8 > 0 else float('inf')

        all_int8_rmse.append(r8)
        all_int4_rmse.append(r4)
        all_int8_snr.append(s8)
        all_int4_snr.append(s4)

        print(f"  {name:<38} {dtype_name:<12} {str(shape):<20} {r8:>10.6f} {r4:>10.6f} {s8:>9.1f} {s4:>9.1f} {ratio:>7.1f}x")

    if all_int8_rmse:
        print()
        print(f"  {'AVERAGE':<38} {'':12} {'':20} {np.mean(all_int8_rmse):>10.6f} {np.mean(all_int4_rmse):>10.6f} {np.mean(all_int8_snr):>9.1f} {np.mean(all_int4_snr):>9.1f} {np.mean(all_int4_rmse)/np.mean(all_int8_rmse):>7.1f}x")

    print()
    print("Interpretation:")
    print("  ratio = int4_RMSE / int8_RMSE  (higher = more degradation from int4)")
    print("  SNR in dB (higher = better quality).  Human-imperceptible threshold: ~30 dB")
    print()
    print("Bandwidth comparison:")
    print("  int8 weights:  1 byte/weight  (current NPU path)")
    print("  int4 weights:  0.5 bytes/weight  (proposed)")
    print("  Theoretical decode speedup:  ~2x (if still BW-bound)")


if __name__ == "__main__":
    main()
