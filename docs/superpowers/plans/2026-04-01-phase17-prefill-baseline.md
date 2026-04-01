# Phase 17: NPU Prefill Baseline Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure NPU prefill throughput at pp=160 using existing decode xclbins, then apply a decision gate to determine whether prefill xclbin investment is warranted.

**Architecture:** Source the decode env, remove the MAX_N=1 decode-only cap, disable Vulkan, and run llama-bench. The XDNA backend's `find_slot()` will select the tile_n=64 slot for N=160 (3 tile columns, ~17% padding). No new kernels or tooling required.

**Tech Stack:** llama-bench, ggml-xdna backend (existing), ~/xclbin-decode/ v1 xclbins (mm.cc-based, tile_n=64)

---

## File Structure

- Create: `docs/phase-17-prefill-baseline.md` — results and decision gate outcome

---

### Task 1: Verify environment prerequisites

**Files:**
- No file changes

- [ ] **Step 1: Confirm xclbins and model exist**

Run:
```bash
ls ~/xclbin-decode/*.xclbin
ls ~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf
ls ./build/bin/llama-bench
```
Expected: four xclbin files (k2048, k4096, k5632, k14336), model file, and binary all present.

- [ ] **Step 2: Confirm decode env loads correctly**

Run:
```bash
source ~/.npu-decode.env && echo "TILE_N=$GGML_XDNA_TILE_N  MIN_N=$GGML_XDNA_MIN_N  XCLBIN=$GGML_XDNA_XCLBIN_PATH"
```
Expected output (values must match):
```
TILE_N=64  MIN_N=1  XCLBIN=/home/corye/xclbin-decode/k2048_n64_decode.xclbin
```

---

### Task 2: Configure environment for prefill measurement

**Files:**
- No file changes

- [ ] **Step 1: Source decode env and apply prefill overrides**

Run the following in a single shell session (all subsequent tasks must run in the same session):
```bash
source ~/.npu-decode.env
unset GGML_XDNA_MAX_N             # remove decode-only N=1 cap
export GGML_XDNA_MIN_N=2          # N=1 decode → CPU; N=160 prefill → NPU
export GGML_VK_VISIBLE_DEVICES="" # disable Vulkan
```

- [ ] **Step 2: Confirm effective configuration**

Run:
```bash
echo "MIN_N=$GGML_XDNA_MIN_N  MAX_N=${GGML_XDNA_MAX_N:-unset}  VK=${GGML_VK_VISIBLE_DEVICES}"
```
Expected:
```
MIN_N=2  MAX_N=unset  VK=
```

---

### Task 3: Run NPU prefill benchmark

**Files:**
- No file changes

- [ ] **Step 1: Run llama-bench (3 reps, pp=160, n=1)**

Run (in the same session as Task 2):
```bash
MODEL=~/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf
./build/bin/llama-bench -m "$MODEL" -p 160 -n 1 -r 3 2>/dev/null
```
Expected: markdown table with two rows per rep — `pp 160` and `tg 1`.

Example of a well-formed output (values will differ):
```
| model         | size   | params | backend | ngl | test   | t/s          |
| llama 8B Q8_0 | 7.95 G | 8.03 B | XDNA    |   0 | pp 160 | X.XX ± Y.YY  |
| llama 8B Q8_0 | 7.95 G | 8.03 B | XDNA    |   0 | tg 1   | Z.ZZ ± W.WW  |
```

- [ ] **Step 2: Record the pp 160 t/s value**

Write down the `pp 160` mean ± stddev from the output. This is the NPU prefill number. The `tg 1` row is CPU decode — not relevant.

- [ ] **Step 3: Apply the decision gate**

Compare the `pp 160` mean against the Vulkan baseline (16.41 t/s):

| Result | Next step |
|--------|-----------|
| ≥ 8 t/s | Phase 17B: build dedicated prefill xclbins (tile_n=128 or 256) |
| 2–8 t/s | bench-prefill.sh sweep across tile_n and pp to find optimal config |
| < 2 t/s | Document as characterization, pause NPU prefill investment |

---

### Task 4: Write results document

**Files:**
- Create: `docs/phase-17-prefill-baseline.md`

- [ ] **Step 1: Create the results document**

Create `docs/phase-17-prefill-baseline.md` with the actual measured values filled in:

```markdown
# Phase 17: NPU Prefill Baseline

**Date:** 2026-04-01
**Model:** Meta-Llama-3-8B-Instruct Q8_0
**Hardware:** Ryzen AI MAX 385 · XDNA2 NPU
**Methodology:** Existing decode xclbins (tile_n=64, mm.cc), MIN_N=2, Vulkan disabled

## Setup

- xclbin: ~/xclbin-decode/ (k2048/k4096/k5632/k14336, tile_n=64)
- GGML_XDNA_MIN_N=2 (N=1 decode → CPU; N=160 prefill → NPU)
- GGML_XDNA_MAX_N: unset (no upper cap)
- GGML_VK_VISIBLE_DEVICES="" (Vulkan disabled)

## Results

| Test | t/s (mean ± stddev) | reps |
|------|---------------------|------|
| NPU prefill pp=160 | [FILL IN] | 3 |
| CPU decode tg=1    | [FILL IN] | 3 |

**Vulkan prefill baseline (Phase 16):** 16.41 t/s  
**NPU/Vulkan ratio:** [FILL IN]×

## Decision Gate Outcome

[State which gate was hit and what the next step is]

## Notes

- tile_n=64 xclbins produce ~17% padding waste at N=160 (3 tile columns).
  A tile_n=128 or 256 xclbin would eliminate this waste.
- This measurement is a lower bound on achievable NPU prefill t/s.
```

- [ ] **Step 2: Fill in all [FILL IN] placeholders with actual measured values**

Replace every `[FILL IN]` with the values from Task 3. No placeholders should remain.

---

### Task 5: Commit

**Files:**
- `docs/phase-17-prefill-baseline.md`

- [ ] **Step 1: Stage and commit**

```bash
git add docs/phase-17-prefill-baseline.md
git commit -m "docs: Phase 17 NPU prefill baseline — [X.XX t/s vs Vulkan 16.41 t/s]

Measured NPU prefill at pp=160 using existing decode xclbins (tile_n=64).
Decision gate outcome: [state outcome here]."
```

Replace `[X.XX t/s vs Vulkan 16.41 t/s]` and the decision gate outcome with actual values.

- [ ] **Step 2: Verify clean working tree**

```bash
git status
```
Expected: `nothing to commit, working tree clean`
