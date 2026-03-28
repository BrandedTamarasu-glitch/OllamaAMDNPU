# Phase 10: bo_a Weight Tile Pre-Staging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-token `memcpy + sync` overhead on weight tiles by caching pre-staged `xrt::bo` objects in `WeightEntry`, so token 2+ dispatches zero weight DMA work.

**Architecture:** Extend `WeightEntry` with an `unordered_map<uint64_t, xrt::bo> tile_bos` keyed by `(slot_idx << 32 | tile_row_index)`. On cache miss (token 1), allocate BO + memcpy + sync once. On cache hit (token 2+), pass the pre-staged BO directly to `dispatch_tile` — no memcpy, no sync. `dispatch_tile` signature changes to accept `xrt::bo &` instead of `const int8_t *` for bo_a.

**Tech Stack:** C++17, XRT (`xrt::bo`, `xrt::kernel`), ggml backend interface

**Spec:** `docs/superpowers/specs/2026-03-28-phase10-design.md`

---

## Chunk 1: Cleanup and Data Structure

### Task 1: Remove probe code

**Files:**
- Modify: `ggml/src/ggml-xdna/ggml-xdna.cpp`

- [ ] **Step 1: Remove `probe_bo_capacity` function and its call**

  In `ggml-xdna.cpp`, delete the entire `probe_bo_capacity` static function (the block beginning with the comment `// Temporary Phase 10 probe:` down through its closing `}`).

  Also remove the call site in `try_init_context`:
  ```cpp
  // DELETE this line:
  probe_bo_capacity(ctx, 0);
  ```

- [ ] **Step 2: Build to confirm clean compile**

  ```bash
  cmake --build build --target ggml-xdna 2>&1 | tail -5
  ```
  Expected: `[100%] Built target ggml-xdna` with no errors.

---

### Task 2: Add `tile_bos` to `WeightEntry`

**Files:**
- Modify: `ggml/src/ggml-xdna/ggml-xdna.cpp` — `WeightEntry` struct (~line 152)

- [ ] **Step 1: Extend `WeightEntry` with the tile BO cache**

  Find the `WeightEntry` struct:
  ```cpp
  struct WeightEntry {
      std::vector<int8_t> quant;   // [M*K] int8
      std::vector<float>  scales;  // [M]   per-row scales
      int64_t cached_m = 0;        // M at quantisation time (for invalidation)
      int64_t cached_k = 0;        // K at quantisation time (for invalidation)
  };
  ```

  Replace with:
  ```cpp
  struct WeightEntry {
      std::vector<int8_t> quant;   // [M*K] int8
      std::vector<float>  scales;  // [M]   per-row scales
      int64_t cached_m = 0;        // M at quantisation time (for invalidation)
      int64_t cached_k = 0;        // K at quantisation time (for invalidation)

      // Pre-staged XRT BOs: one per (slot_idx, tile_row_index).
      // key   = ((uint64_t)slot_idx << 32) | (uint64_t)(m0 / tile_m)
      // value = host_only xrt::bo, memcpy'd and sync'd once at first use.
      // Destroyed automatically when this entry is erased (M/K dimension change).
      std::unordered_map<uint64_t, xrt::bo> tile_bos;
  };
  ```

- [ ] **Step 2: Build to confirm clean compile**

  ```bash
  cmake --build build --target ggml-xdna 2>&1 | tail -5
  ```
  Expected: `[100%] Built target ggml-xdna` — no errors.

---

## Chunk 2: dispatch_tile and mul_mat

### Task 3: Update `dispatch_tile` to accept pre-staged bo_a

**Files:**
- Modify: `ggml/src/ggml-xdna/ggml-xdna.cpp` — `dispatch_tile` function (~line 248)

  Apply the following as a single replacement (both the signature change and the kernel call update in one edit — do not split into two passes):

- [ ] **Step 1: Replace the entire dispatch_tile function header through the kernel call**

  Find (verbatim, from signature through kernel call):
  ```cpp
  static bool dispatch_tile(ggml_backend_xdna_context * ctx,
                            const int8_t * a_tile,
                            const int8_t * b_tile,
                            int32_t      * c_tile,
                            int           slot_idx) {
      auto & s          = ctx->slots[slot_idx];
      auto & kern       = *s.kernel;
      auto & instr      = s.bo_instr;
      auto & bo_a       = s.bo_a;
      auto & bo_b       = s.bo_b;
      auto & bo_c       = s.bo_c;
      auto & bo_tmp     = s.bo_tmp;
      auto & bo_trace   = s.bo_trace;
      auto & instr_data = s.instr_data;

      const int64_t tm = s.tile_m;
      const int64_t tk = s.tile_k;
      const int64_t tn = s.tile_n;

      std::memcpy(bo_a.map<int8_t *>(), a_tile, (size_t)tm * tk * sizeof(int8_t));
      std::memcpy(bo_b.map<int8_t *>(), b_tile, (size_t)tk * tn * sizeof(int8_t));

      bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      constexpr unsigned int opcode = 3;
      auto run = kern(opcode, instr,
                      static_cast<uint32_t>(instr_data.size()), // safe: load_instr_file enforces 4 MB / 4 = 1M word limit
                      bo_a, bo_b, bo_c, bo_tmp, bo_trace);
  ```

  Replace with:
  ```cpp
  static bool dispatch_tile(ggml_backend_xdna_context * ctx,
                            xrt::bo      & bo_a_staged,
                            const int8_t * b_tile,
                            int32_t      * c_tile,
                            int           slot_idx) {
      auto & s          = ctx->slots[slot_idx];
      auto & kern       = *s.kernel;
      auto & instr      = s.bo_instr;
      auto & bo_b       = s.bo_b;
      auto & bo_c       = s.bo_c;
      auto & bo_tmp     = s.bo_tmp;
      auto & bo_trace   = s.bo_trace;
      auto & instr_data = s.instr_data;

      const int64_t tm = s.tile_m;   // still needed for bo_c memcpy in function tail
      const int64_t tk = s.tile_k;
      const int64_t tn = s.tile_n;

      // bo_a_staged is pre-staged (memcpy + sync done once at cache-miss time).
      // Only bo_b (activations) changes per token — copy and sync it here.
      std::memcpy(bo_b.map<int8_t *>(), b_tile, (size_t)tk * tn * sizeof(int8_t));
      bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      constexpr unsigned int opcode = 3;
      auto run = kern(opcode, instr,
                      static_cast<uint32_t>(instr_data.size()), // safe: load_instr_file enforces 4 MB / 4 = 1M word limit
                      bo_a_staged, bo_b, bo_c, bo_tmp, bo_trace);
  ```

- [ ] **Step 2: Build to confirm dispatch_tile compiles (call sites will error — expected)**

  ```bash
  cmake --build build --target ggml-xdna 2>&1 | tail -10
  ```
  Expected: errors about `dispatch_tile` call sites only. Confirm the errors say something like "no matching function" or "cannot convert" at the call sites, not inside `dispatch_tile` itself.

---

### Task 4: Update `mul_mat` to use the tile BO cache

**Files:**
- Modify: `ggml/src/ggml-xdna/ggml-xdna.cpp` — `ggml_backend_xdna_mul_mat`, inner m0 loop (~line 410)

  Note: `sl` is declared as `const auto & sl = ctx->slots[slot_idx]` — this does **not** need to change. `xrt::kernel::group_id()` is a const method; `unique_ptr::operator->() const` returns a non-const `element_type *`, so calling `sl.kernel->group_id(3)` compiles without removing `const`.

- [ ] **Step 1: Replace the inner m0 loop body with cache-aware dispatch**

  Find (the section that fills `tile_a` and calls `dispatch_tile`):
  ```cpp
          for (int64_t m0 = 0; m0 < M; m0 += tm) {
              // Fill tile_a from cached qa[m0:m0+tm, 0:K].
              // Zero-fill partial tiles only; full tiles (m0+tm <= M) have all tm rows
              // written by the memcpy loop below (K bytes each), so pre-clearing would be redundant.
              if ((m0 + tm) > M) { std::fill(ctx->tile_a.begin(), ctx->tile_a.end(), 0); }
              for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                  std::memcpy(ctx->tile_a.data() + mi * tk,
                              qa + (m0 + mi) * K,
                              K); // K == tk (supports_op guarantee)
              }

              if (!dispatch_tile(ctx, ctx->tile_a.data(), ctx->tile_b.data(), ctx->tile_c.data(), slot_idx)) {
                  return false;
              }
  ```

  Replace with:
  ```cpp
          for (int64_t m0 = 0; m0 < M; m0 += tm) {
              // Cache key encodes both slot and tile row to handle multi-slot setups.
              const uint64_t tile_key = ((uint64_t)slot_idx << 32) |
                                        (uint64_t)(m0 / tm);
              auto it = entry.tile_bos.find(tile_key);
              if (it == entry.tile_bos.end()) {
                  // Cache miss (first token): fill tile_a, allocate BO, memcpy + sync once.
                  if ((m0 + tm) > M) { std::fill(ctx->tile_a.begin(), ctx->tile_a.end(), 0); }
                  for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                      std::memcpy(ctx->tile_a.data() + mi * tk,
                                  qa + (m0 + mi) * K,
                                  K); // K == tk (supports_op guarantee)
                  }
                  xrt::bo new_bo(ctx->device,
                                 (size_t)tm * tk * sizeof(int8_t),
                                 xrt::bo::flags::host_only,
                                 sl.kernel->group_id(3));
                  std::memcpy(new_bo.map<int8_t *>(), ctx->tile_a.data(), (size_t)tm * tk);
                  new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                  entry.tile_bos.emplace(tile_key, std::move(new_bo));
                  it = entry.tile_bos.find(tile_key);  // re-acquire after emplace (invalidates iterators)
              }
              // Cache hit: bo is pre-staged — no tile_a fill, no memcpy, no sync.

              if (!dispatch_tile(ctx, it->second, ctx->tile_b.data(), ctx->tile_c.data(), slot_idx)) {
                  return false;
              }
  ```

- [ ] **Step 2: Build — should now compile cleanly**

  ```bash
  cmake --build build --target ggml-xdna 2>&1 | tail -5
  ```
  Expected: `[100%] Built target ggml-xdna` — no errors, no warnings.

- [ ] **Step 3: Also build llama-bench**

  ```bash
  cmake --build build --target llama-bench 2>&1 | tail -3
  ```
  Expected: `[100%] Built target llama-bench`

---

## Chunk 3: Verification

### Task 5: Benchmark and verify improvement

The primary verification is across `llama-bench` repetitions (`-r 3`): run 1 = all cache misses (cold, same speed as today); runs 2–3 = all cache hits (warm, should be faster). This works for both prefill and decode because each repetition re-uses the same weight tiles.

The spec calls for decode verification (MIN_N=1, `-p 0 -n 64`). If the active `.zshrc` config is Phase 6 (MIN_N=2, prefill only), also run the prefill bench below to confirm the cache works — then configure for decode to match the spec.

- [ ] **Step 1: Run 3-repetition prefill bench — confirm run 2+ faster than run 1**

  ```bash
  cd build/bin
  ./llama-bench -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p 512 -n 0 -r 3 --no-warmup 2>&1
  ```

  Record pp512 t/s for all 3 runs. Run 1 = cold (cache miss on every tile). Runs 2–3 = warm (cache hit on every tile). Expect a measurable improvement from run 1 to run 2.

- [ ] **Step 2: Verify correctness**

  ```bash
  ./llama-cli -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p "The capital of France is" -n 10 2>&1 | tail -5
  ```
  Expected: coherent completion ("Paris" or similar). If output is garbage or NaN, the pre-staged BO content is wrong — check the memcpy in the cache-miss path.

- [ ] **Step 3: Run decode bench if decode xclbins are available**

  If `~/xclbin-decode/` contains decode xclbins, enable decode mode:
  ```bash
  # Temporarily override for the bench (don't modify .zshrc yet)
  env GGML_XDNA_MIN_N=1 GGML_XDNA_MAX_N=1 \
    ./llama-bench -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p 0 -n 64 -r 3 --no-warmup 2>&1
  ```
  Expected: tg64 t/s for runs 2–3 measurably faster than run 1 (and faster than the current 0.65 t/s baseline).

  If decode xclbins are not loaded/configured, skip this step — the prefill bench in Step 1 is sufficient to confirm the cache mechanism works.

- [ ] **Step 4: Commit**

  ```bash
  git add ggml/src/ggml-xdna/ggml-xdna.cpp
  git commit -m "feat: Phase 10 — pre-stage bo_a weight tiles to eliminate per-token DMA

  Add tile_bos cache (unordered_map<uint64_t, xrt::bo>) to WeightEntry.
  On first dispatch of each (slot_idx, tile_row) pair, allocate host_only
  xrt::bo, memcpy weight tile, sync once. On subsequent tokens, pass the
  pre-staged BO directly — no memcpy, no sync for weight tiles.

  dispatch_tile now accepts xrt::bo& instead of const int8_t* for bo_a.
  bo_b (activations) still copied and synced per dispatch as before.

  Also removes temporary probe_bo_capacity function added during design.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Validation Checklist (complete before claiming done)

- [ ] `cmake --build build --target ggml-xdna llama-bench` — zero errors, zero warnings
- [ ] `llama-bench -r 3`: pp512 t/s run 2 and run 3 measurably faster than run 1
- [ ] `llama-cli` short generation produces coherent output (no garbage / NaN)
- [ ] `git diff HEAD~1` reviewed — no unintended changes outside `ggml-xdna.cpp`
