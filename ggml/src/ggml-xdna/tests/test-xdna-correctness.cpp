// test-xdna-correctness.cpp
// Validates the quant->int8 matmul->dequant pipeline used by ggml-xdna
// against a CPU float reference.  Does not require XRT or a loaded xclbin.

#include "ggml-xdna-quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ---- CPU float reference matmul: C[M,N] = A[M,K] * B[K,N] ----------------

static void ref_matmul(const float * A, const float * B, float * C,
                       int64_t M, int64_t K, int64_t N) {
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

// ---- Tiled pipeline simulation (mirrors ggml_backend_xdna_mul_mat) --------
// A[M,K] (weights, row-major), B_src[N,K] (activations stored as [N,K]).
// Tiles over M, N, K with zero-padding at edges — exercising the same paths
// as the production code including the B-matrix transpose per tile.
static void xdna_pipeline(const float * A_f32, const float * B_f32,
                           float * C_out,
                           int64_t M, int64_t K, int64_t N,
                           int64_t tm, int64_t tk, int64_t tn) {
    // Quantise A (weights) [M,K]
    std::vector<int8_t> qA(M * K);
    std::vector<float>  sA(M);
    quant_f32_to_int8(A_f32, qA.data(), M, K, sA.data());

    // Quantise B (activations) [N,K]
    std::vector<int8_t> qB(N * K);
    std::vector<float>  sB(N);
    quant_f32_to_int8(B_f32, qB.data(), N, K, sB.data());

    std::fill(C_out, C_out + M * N, 0.0f);

    const int64_t tile_a_sz = tm * tk;
    const int64_t tile_b_sz = tk * tn;
    const int64_t tile_c_sz = tm * tn;

    std::vector<int8_t>  tile_a(tile_a_sz, 0);
    std::vector<int8_t>  tile_b(tile_b_sz, 0);
    std::vector<int32_t> tile_c(tile_c_sz, 0);
    std::vector<int32_t> acc(tile_c_sz, 0);

    for (int64_t m0 = 0; m0 < M; m0 += tm) {
        for (int64_t n0 = 0; n0 < N; n0 += tn) {
            std::fill(acc.begin(), acc.end(), 0);

            for (int64_t k0 = 0; k0 < K; k0 += tk) {
                // Copy + zero-pad tile_a from qA[m0:m0+tm, k0:k0+tk]
                std::fill(tile_a.begin(), tile_a.end(), 0);
                for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                    const int64_t cols = std::min(tk, K - k0);
                    for (int64_t ki = 0; ki < cols; ki++) {
                        tile_a[mi * tk + ki] = qA[(m0 + mi) * K + k0 + ki];
                    }
                }

                // Copy + zero-pad tile_b from qB[n0:n0+tn, k0:k0+tk],
                // transposed into [K,N] layout (tile_b[ki,ni] = qB[n0+ni, k0+ki]).
                std::fill(tile_b.begin(), tile_b.end(), 0);
                for (int64_t ni = 0; ni < tn && (n0 + ni) < N; ni++) {
                    for (int64_t ki = 0; ki < tk && (k0 + ki) < K; ki++) {
                        tile_b[ki * tn + ni] = qB[(n0 + ni) * K + (k0 + ki)];
                    }
                }

                // Simulate kernel: tile_c[m,n] = sum_k tile_a[m,k] * tile_b[k,n]
                std::fill(tile_c.begin(), tile_c.end(), 0);
                for (int64_t mi = 0; mi < tm; mi++) {
                    for (int64_t ni = 0; ni < tn; ni++) {
                        int32_t dot = 0;
                        for (int64_t ki = 0; ki < tk; ki++) {
                            dot += static_cast<int32_t>(tile_a[mi * tk + ki]) *
                                   static_cast<int32_t>(tile_b[ki * tn + ni]);
                        }
                        tile_c[mi * tn + ni] = dot;
                    }
                }

                for (int64_t i = 0; i < tile_c_sz; i++) {
                    acc[i] += tile_c[i];
                }
            }

            // Scatter with dequantisation
            for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                for (int64_t ni = 0; ni < tn && (n0 + ni) < N; ni++) {
                    C_out[(m0 + mi) * N + (n0 + ni)] =
                        static_cast<float>(acc[mi * tn + ni])
                        * sA[m0 + mi] * sB[n0 + ni];
                }
            }
        }
    }
}

// ---- Test harness ----------------------------------------------------------

static float max_abs_error(const float * a, const float * b, int64_t n) {
    float e = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        e = std::max(e, std::fabs(a[i] - b[i]));
    }
    return e;
}

static int test_tiled_pipeline(const char * name,
                         int64_t M, int64_t K, int64_t N,
                         int64_t tile_m, int64_t tile_k, int64_t tile_n,
                         float tol) {
    printf("Test %-12s [M=%lld K=%lld N=%lld tile=%lldx%lldx%lld] ...\n",
           name, (long long)M, (long long)K, (long long)N,
           (long long)tile_m, (long long)tile_k, (long long)tile_n);

    std::vector<float> A(M * K);
    std::vector<float> B(N * K);
    for (int64_t i = 0; i < M * K; i++) {
        A[i] = static_cast<float>((i % 17) - 8) * 0.25f;
    }
    for (int64_t i = 0; i < N * K; i++) {
        B[i] = static_cast<float>((i % 13) - 6) * 0.1f;
    }

    // CPU float reference: C_ref[M,N] = A[M,K] * B^T[K,N]
    // B stored as [N,K]; transpose to [K,N] for ref_matmul.
    std::vector<float> B_t(K * N);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k++) {
            B_t[k * N + n] = B[n * K + k];
        }
    }
    std::vector<float> C_ref(M * N, 0.0f);
    ref_matmul(A.data(), B_t.data(), C_ref.data(), M, K, N);

    // Tiled pipeline (mirrors production backend)
    std::vector<float> C_npu(M * N, 0.0f);
    xdna_pipeline(A.data(), B.data(), C_npu.data(), M, K, N, tile_m, tile_k, tile_n);

    const float err = max_abs_error(C_ref.data(), C_npu.data(), M * N);
    printf("  max_abs_error = %.6f  (tol=%.6f)\n", err, tol);
    if (err > tol) {
        fprintf(stderr, "  FAIL: error %.6f exceeds tolerance %.6f\n", err, tol);
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

// Test that NaN/Inf rows produce all-zero output (no UB, no partial corruption).
static int test_nan_inf_quant() {
    constexpr int64_t rows = 3, cols = 4;
    float src[rows * cols] = {
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),  std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),  std::numeric_limits<float>::infinity(),
        1.0f, std::numeric_limits<float>::quiet_NaN(), -2.0f, std::numeric_limits<float>::infinity(),
    };
    int8_t dst[rows * cols] = {};
    float  scales[rows]     = {};
    quant_f32_to_int8(src, dst, rows, cols, scales);

    int failures = 0;
    // Rows 0 (all-NaN) and 1 (all-Inf) must be fully zeroed.
    for (int r = 0; r < 2; r++) {
        if (scales[r] != 0.0f) {
            fprintf(stderr, "  FAIL nan_inf_quant: row %d scale=%f, want 0\n", r, (double)scales[r]);
            failures++;
        }
        for (int c = 0; c < cols; c++) {
            if (dst[r * cols + c] != 0) {
                fprintf(stderr, "  FAIL nan_inf_quant: row %d col %d dst=%d, want 0\n", r, c, dst[r*cols+c]);
                failures++;
            }
        }
    }
    // Row 2 (mixed): finite values 1.0 and -2.0 should be quantised; NaN/Inf mapped to 0.
    if (scales[2] <= 0.0f) {
        fprintf(stderr, "  FAIL nan_inf_quant: row 2 scale=%f, want > 0\n", (double)scales[2]);
        failures++;
    }
    if (failures == 0) { printf("Test nan_inf_quant ... PASS\n"); }
    return failures;
}

int main() {
    int failures = 0;

    failures += test_nan_inf_quant();

    // Tile-aligned cases
    failures += test_tiled_pipeline("small",     4,  4,  4,  4,  4,  4, 0.25f);
    failures += test_tiled_pipeline("tile",     32, 32, 32, 32, 32, 32, 0.25f);
    failures += test_tiled_pipeline("rect",      8, 64, 16, 32, 32, 32, 0.40f);

    // Non-tile-aligned cases — exercise zero-padding paths and B-transpose edge conditions
    failures += test_tiled_pipeline("unaligned", 33, 37, 17, 32, 32, 32, 0.50f);
    failures += test_tiled_pipeline("odd-K",     16, 65,  8, 32, 32, 32, 0.50f);

    if (failures > 0) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("\nAll correctness tests PASSED\n");
    return 0;
}
