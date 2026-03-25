/**
 * test-xdna-correctness.cpp — unit tests for ggml-xdna-quant.h
 *
 * Tests the quantisation helper in isolation — no XRT, no NPU, no ggml dependency.
 * Compile and run standalone:
 *
 *   g++ -std=c++17 -I. -o test-xdna-correctness test-xdna-correctness.cpp && ./test-xdna-correctness
 *
 * All tests use a simple pass/fail framework with no external test library.
 */

#include "ggml-xdna-quant.h"

#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT_TRUE(cond)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                             \
        } else {                                                                 \
            g_pass++;                                                             \
        }                                                                        \
    } while (0)

#define EXPECT_NEAR(a, b, tol) EXPECT_TRUE(std::fabs((a) - (b)) <= (tol))

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float dequant(int8_t q, float scale) {
    return static_cast<float>(q) * scale;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Basic single-row quantisation: values round-trip within expected error.
static void test_basic_quantisation() {
    const int64_t cols = 4;
    float src[4]  = { 1.0f, -1.0f, 0.5f, -0.5f };
    int8_t dst[4] = {};
    float scales[1] = {};

    quant_f32_to_int8(src, dst, 1, cols, scales);

    // scale = max(|src|) / 127 = 1.0 / 127
    const float expected_scale = 1.0f / 127.0f;
    EXPECT_NEAR(scales[0], expected_scale, 1e-6f);

    // Each value should round-trip within 1 ULP of scale.
    for (int i = 0; i < cols; i++) {
        const float reconstructed = dequant(dst[i], scales[0]);
        EXPECT_NEAR(reconstructed, src[i], expected_scale);
    }
}

// All-zero row: output is all zeros, scale is zero.
static void test_zero_row() {
    const int64_t cols = 4;
    float src[4]   = { 0.0f, 0.0f, 0.0f, 0.0f };
    int8_t dst[4]  = { 99, 99, 99, 99 }; // pre-fill with non-zero to detect correctness
    float scales[1] = { 99.0f };

    quant_f32_to_int8(src, dst, 1, cols, scales);

    EXPECT_NEAR(scales[0], 0.0f, 1e-10f);
    for (int i = 0; i < cols; i++) {
        EXPECT_TRUE(dst[i] == 0);
    }
}

// All-NaN row: should not produce UB; output is all zeros, scale is zero.
static void test_nan_row() {
    const int64_t cols = 4;
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    float src[4]   = { nan_val, nan_val, nan_val, nan_val };
    int8_t dst[4]  = { 1, 2, 3, 4 };
    float scales[1] = { 1.0f };

    quant_f32_to_int8(src, dst, 1, cols, scales);

    EXPECT_NEAR(scales[0], 0.0f, 1e-10f);
    for (int i = 0; i < cols; i++) {
        EXPECT_TRUE(dst[i] == 0);
    }
}

// All-Inf row: should not produce UB; output is all zeros, scale is zero.
static void test_inf_row() {
    const float inf = std::numeric_limits<float>::infinity();
    float src[4]   = { inf, -inf, inf, -inf };
    int8_t dst[4]  = { 1, 2, 3, 4 };
    float scales[1] = { 1.0f };

    quant_f32_to_int8(src, dst, 1, 4, scales);

    EXPECT_NEAR(scales[0], 0.0f, 1e-10f);
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(dst[i] == 0);
    }
}

// Mixed NaN/Inf with finite values: finite values quantise correctly.
static void test_mixed_nan_finite() {
    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    float src[4]  = { nan_val, 2.0f, nan_val, -1.0f };
    int8_t dst[4] = {};
    float scales[1] = {};

    quant_f32_to_int8(src, dst, 1, 4, scales);

    // max_abs over finite values = 2.0f; scale = 2.0/127
    const float expected_scale = 2.0f / 127.0f;
    EXPECT_NEAR(scales[0], expected_scale, 1e-6f);

    // NaN elements are treated as 0 during quantisation
    EXPECT_TRUE(dst[0] == 0);
    EXPECT_NEAR(dequant(dst[1], scales[0]), 2.0f, expected_scale);
    EXPECT_TRUE(dst[2] == 0);
    EXPECT_NEAR(dequant(dst[3], scales[0]), -1.0f, expected_scale);
}

// Negative values: asymmetric rows quantise correctly (symmetric scheme).
static void test_negative_values() {
    float src[4]   = { -4.0f, -2.0f, -1.0f, 0.0f };
    int8_t dst[4]  = {};
    float scales[1] = {};

    quant_f32_to_int8(src, dst, 1, 4, scales);

    const float expected_scale = 4.0f / 127.0f;
    EXPECT_NEAR(scales[0], expected_scale, 1e-6f);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(dequant(dst[i], scales[0]), src[i], expected_scale);
    }
}

// Multi-row: each row gets an independent scale.
static void test_multi_row() {
    // Row 0: max 1.0, Row 1: max 10.0, Row 2: max 100.0
    float src[12] = {
        1.0f, -1.0f,  0.0f, 0.5f,
        10.0f, -5.0f, 3.0f, 0.0f,
        100.0f, 50.0f, -25.0f, 0.0f
    };
    int8_t dst[12] = {};
    float scales[3] = {};

    quant_f32_to_int8(src, dst, 3, 4, scales);

    EXPECT_NEAR(scales[0],   1.0f / 127.0f, 1e-5f);
    EXPECT_NEAR(scales[1],  10.0f / 127.0f, 1e-4f);
    EXPECT_NEAR(scales[2], 100.0f / 127.0f, 1e-3f);

    // Round-trip check for each row.
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            EXPECT_NEAR(dequant(dst[r * 4 + c], scales[r]), src[r * 4 + c], scales[r]);
        }
    }
}

// Clamping: values beyond ±127 quant levels must not overflow int8.
static void test_clamping() {
    // All values identical — quantised to ±127.
    float src[4]   = { 5.0f, 5.0f, -5.0f, -5.0f };
    int8_t dst[4]  = {};
    float scales[1] = {};

    quant_f32_to_int8(src, dst, 1, 4, scales);

    // max maps to 127, min maps to -127.
    EXPECT_TRUE(dst[0] == 127);
    EXPECT_TRUE(dst[1] == 127);
    EXPECT_TRUE(dst[2] == -127);
    EXPECT_TRUE(dst[3] == -127);
}

// Cache invalidation simulation: verify that re-quantising with different M/K
// produces correct results (correctness of the quantisation function itself;
// the cache invalidation logic lives in ggml-xdna.cpp).
static void test_requantise_different_shape() {
    // Shape A: 2 rows × 4 cols
    float src_a[8]  = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f };
    int8_t dst_a[8] = {};
    float scales_a[2] = {};
    quant_f32_to_int8(src_a, dst_a, 2, 4, scales_a);

    // Shape B (same pointer address scenario): 1 row × 8 cols
    float src_b[8]  = { 1.0f, 0.5f, -0.5f, -1.0f, 2.0f, -2.0f, 0.25f, -0.25f };
    int8_t dst_b[8] = {};
    float scales_b[1] = {};
    quant_f32_to_int8(src_b, dst_b, 1, 8, scales_b);

    // Results should match independent quantisation — no cross-contamination.
    const float expected_scale_b = 2.0f / 127.0f;
    EXPECT_NEAR(scales_b[0], expected_scale_b, 1e-5f);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(dequant(dst_b[i], scales_b[0]), src_b[i], expected_scale_b);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    test_basic_quantisation();
    test_zero_row();
    test_nan_row();
    test_inf_row();
    test_mixed_nan_finite();
    test_negative_values();
    test_multi_row();
    test_clamping();
    test_requantise_different_shape();

    std::printf("\n%d passed  |  %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
