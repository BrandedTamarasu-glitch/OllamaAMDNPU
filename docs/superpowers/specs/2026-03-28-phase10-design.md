# Phase 10 Design: bo_a Weight Tile Pre-Staging

**Date:** 2026-03-28
**Model:** Meta-Llama-3.1-8B-Instruct Q4_K_M
**Hardware:** Ryzen AI MAX 385 · XDNA2 NPU
**Success criterion:** Any measurable NPU decode improvement on token 2+ vs token 1

---

## Problem

Every call to `dispatch_tile` does the following for the weight tile (bo_a):

1. `memcpy(bo_a.map(), tile_a_data, tile_m * tile_k)` — copy quantised weight tile to XRT BO
2. `bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE)` — flush CPU cache (host_only BO)

Weight matrices are stable across decode tokens. The same `(src0_ptr, m0)` tile is copied and flushed identically on every token — pure overhead.

With tile_m=2048, tile_k=2048: bo_a = 4 MB per tile. For ~704 dispatch_tile calls per decode token, this is the dominant cost at 0.65 t/s current decode speed.

---

## Probe Results (2026-03-28)

```
slot 0: tile 2048×2048×64  →  bo_a = 4,194,304 bytes (4 MB)
XRT BO probe: allocated 2048 BOs = 8,192 MB — hit probe limit, no XRT failure
```

`host_only` BOs live in host DRAM (30 GB available). XRT imposes no meaningful count or size limit. `sync(TO_DEVICE)` is a CPU cache flush, not an NPU SRAM copy. Pre-staging is safe and memory-pressure-free.

---

## Design

### Approach: Lazy BO Cache per Weight Tile

Extend `WeightEntry` with a map of pre-staged `xrt::bo` objects. On first dispatch of a given `(slot_idx, tile_row)`, allocate a BO, copy the tile data, sync once, and store it. On all subsequent tokens, use the stored BO directly — no memcpy, no sync.

**Strategy: lazy warm-up.** Token 1 builds the cache (same performance as today). Token 2+ has zero memcpy/sync overhead on all weight tiles.

### Data Structures

```cpp
struct WeightEntry {
    std::vector<int8_t> quant;          // [M*K] int8 — unchanged
    std::vector<float>  scales;          // [M]   per-row — unchanged
    int64_t cached_m = 0;               // unchanged
    int64_t cached_k = 0;               // unchanged

    // NEW: pre-staged XRT BOs, one per (slot_idx, tile_row_index).
    // key   = ((uint64_t)slot_idx << 32) | (uint64_t)(m0 / tile_m)
    // value = host_only xrt::bo, data copied and synced once at first use
    std::unordered_map<uint64_t, xrt::bo> tile_bos;
};
```

### dispatch_tile Signature Change

```cpp
// Before:
static bool dispatch_tile(ggml_backend_xdna_context * ctx,
                          const int8_t * a_tile,
                          const int8_t * b_tile,
                          int32_t      * c_tile,
                          int           slot_idx);

// After:
static bool dispatch_tile(ggml_backend_xdna_context * ctx,
                          xrt::bo      & bo_a_staged,   // pre-staged; no memcpy/sync
                          const int8_t * b_tile,
                          int32_t      * c_tile,
                          int           slot_idx);
```

The slot's `XclbinSlot::bo_a` member is kept (used during slot init for group_id resolution) but no longer written during dispatch.

### mul_mat m0 Loop

```
for each n0:
    fill tile_b (activations — always fresh)

    for each m0:
        tile_key = ((uint64_t)slot_idx << 32) | (uint64_t)(m0 / tile_m)

        if tile_key NOT in entry.tile_bos:           // cache miss (token 1)
            fill tile_a from entry.quant[m0:]        // existing code, unchanged
            allocate xrt::bo(device, tile_m*tile_k,
                             host_only, kernel.group_id(3))
            memcpy tile_a → new_bo
            new_bo.sync(TO_DEVICE)
            entry.tile_bos[tile_key] = std::move(new_bo)

        dispatch_tile(ctx, entry.tile_bos[tile_key], tile_b, tile_c, slot_idx)
        scatter tile_c into dst
```

### Invalidation

The existing stale-entry check erases the full `WeightEntry` when `(M, K)` changes for a `src0->data` pointer. Since `tile_bos` lives inside `WeightEntry`, all pre-staged BOs for that weight are destroyed automatically — no additional invalidation logic required.

---

## What Changes

| Location | Change |
|---|---|
| `WeightEntry` | Add `tile_bos` map |
| `dispatch_tile` | Accept `xrt::bo &` instead of `const int8_t *` for bo_a; remove internal memcpy+sync of bo_a |
| `mul_mat` m0 loop | Add tile_key lookup; cache-miss path allocates+stages BO; cache-hit path skips tile_a fill, memcpy, sync |
| `try_init_context` | Remove temporary probe (`probe_bo_capacity` call + function) |

No new files. No changes to slot init, env parsing, supports_op, graph_compute, or the device/registry interface.

---

## What Does NOT Change

- **bo_b (activations):** copied and synced on every dispatch — activations are different every token
- **bo_c, bo_tmp, bo_trace:** unchanged
- **Slot init:** `XclbinSlot::bo_a` allocated as before; group_id(3) still used when creating new tile BOs
- **Prefill path (N > 1):** same weight tiles, same miss/hit logic — pre-staging works identically for prefill but provides less benefit since prefill doesn't repeat tokens

---

## Verification Plan

1. Build and run `llama-bench -p 0 -n 64 -r 3` in NPU-decode mode (MIN_N=1, decode xclbins loaded)
2. Token 1 decode speed should match current ~0.65 t/s (cache miss, identical to today)
3. Token 2+ decode speed should be measurably faster (cache hit, no bo_a overhead)
4. Run 3 repetitions — all repetitions after the first should show improved speed
5. Confirm no correctness regression vs CPU output

---

## Open Questions

- Decode xclbin tile dimensions: probe was run on prefill xclbins (tile_m=2048). Decode xclbins (~/xclbin-decode/) may have different tile_m/tile_k — verify before assuming 4 MB/tile applies to decode.
- If decode tile_m < 2048 (e.g., 32), per-tile BO overhead is smaller but the cache map key packing remains correct.
