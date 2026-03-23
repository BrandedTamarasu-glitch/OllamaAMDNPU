// ggml-xdna-quant.h — symmetric int8 quantisation helper (no XRT dependency).
// Shared between ggml-xdna.cpp (production) and test-xdna-correctness.cpp (tests).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Symmetric per-row int8 quantisation.
// Each row r: scale = max(|row|) / 127; value = clamp(round(x / scale), -127, 127).
// Clamped to [-127, 127] (symmetric range; avoids -128 which has no positive counterpart).
// Non-finite elements (NaN/Inf) are skipped during the max-abs scan; a row whose
// finite maximum is 0 (or all-NaN) is zeroed out with scale=0 to avoid UB in the
// static_cast<int32_t> that follows (C++17 §7.10p1: float→int UB when out of range).
// float = int8 * scale.
static inline void quant_f32_to_int8(const float * src, int8_t * dst,
                                     int64_t rows, int64_t cols, float * scales) {
    for (int64_t r = 0; r < rows; r++) {
        const float * row_src = src + r * cols;
        int8_t      * row_dst = dst + r * cols;
        // Rows with max(|finite values|) below this threshold are treated as zero to
        // avoid scale underflow and subsequent UB in the float→int cast (C++17 §7.10p1).
        constexpr float kQuantZeroThreshold = 1e-8f;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < cols; i++) {
            const float v = row_src[i];
            if (std::isfinite(v)) {
                max_abs = std::max(max_abs, std::fabs(v));
            }
        }
        if (max_abs < kQuantZeroThreshold) {
            // All-zero, all-NaN, or all-Inf row — zero out safely.
            std::fill(row_dst, row_dst + cols, int8_t(0));
            scales[r] = 0.0f;
            continue;
        }
        scales[r] = max_abs / 127.0f;
        const float inv_scale = 127.0f / max_abs;
        for (int64_t i = 0; i < cols; i++) {
            const float v = std::isfinite(row_src[i]) ? row_src[i] : 0.0f;
            int32_t q = static_cast<int32_t>(std::round(v * inv_scale));
            row_dst[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
    }
}
