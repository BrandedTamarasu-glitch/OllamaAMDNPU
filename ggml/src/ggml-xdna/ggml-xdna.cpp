/**
 * ggml-xdna.cpp — ggml backend for AMD XDNA2 NPU (RyzenAI-npu5)
 *
 * Architecture:
 *   - Handles GGML_OP_MUL_MAT by tiling and dispatching to the NPU via XRT.
 *   - All other ops fall back to the CPU backend automatically because
 *     supports_op() returns false for them.
 *   - Uses the mlir-aie xclbin dispatch model: a compiled xclbin encodes a
 *     fixed tile size (TILE_M × TILE_K × TILE_N); we tile larger matrices
 *     at the host level and call the kernel once per tile.
 *
 * Kernel contract (matches mlir-aie matrix_multiplication int8 build):
 *   The MLIR_AIE kernel is invoked as:
 *     kernel(opcode, bo_instr, instr_count, bo_a, bo_b, bo_c, bo_tmp, bo_trace)
 *   where:
 *     opcode     = 3 (standard DPU execute opcode)
 *     bo_instr   = instruction buffer loaded from _sequence.bin
 *     instr_count= number of 32-bit instruction words
 *     bo_a       = int8 input  [TILE_M × TILE_K]
 *     bo_b       = int8 input  [TILE_K × TILE_N]  (column-major / transposed)
 *     bo_c       = int8/int16/int32 output [TILE_M × TILE_N]
 *     bo_tmp     = scratch buffer (1 byte minimum)
 *     bo_trace   = trace buffer  (1 byte minimum, unused)
 *
 *   Matrix dimensions are baked into the xclbin at compile time.
 *   Use GGML_XDNA_TILE_M / _K / _N to tell the backend what tile size
 *   the xclbin was compiled for (defaults: 32 × 32 × 32).
 *
 * Environment variables:
 *   GGML_XDNA_XCLBIN_PATH — path to compiled xclbin  (required to dispatch)
 *   GGML_XDNA_INSTR_PATH  — path to _sequence.bin    (required to dispatch)
 *   GGML_XDNA_TILE_M      — tile rows    (default 32, range 1–1024)
 *   GGML_XDNA_TILE_K      — tile inner   (default 32, range 1–1024)
 *   GGML_XDNA_TILE_N      — tile cols    (default 32, range 1–1024)
 *   GGML_XDNA_MIN_BATCH   — minimum M×N×K to offload (default 32768)
 *
 * Build the xclbin with:
 *   cd mlir-aie/programming_examples/basic/matrix_multiplication/single_core
 *   make dtype_in=i8 dtype_out=i32 AIE_TARGET=aie2p \
 *        M=32 K=32 N=32 m=32 k=32 n=32
 *   # produces: build/single_core_32x32x32_32x32x32.xclbin
 *   #           build/single_core_sequence.bin
 */

#include "ggml-xdna.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>

#include "ggml-xdna-quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct ggml_backend_xdna_context {
    xrt::device                      device;
    std::unique_ptr<xrt::hw_context> hw_ctx;
    std::unique_ptr<xrt::kernel>     matmul_kernel;

    // Instruction buffer (loaded from _sequence.bin).
    // Synced to device once in try_init_context; never re-synced because the
    // buffer is read-only after init.
    std::vector<uint32_t> instr_data;
    xrt::bo               bo_instr;

    // Pre-allocated data buffers (created once in try_init_context).
    xrt::bo bo_a;
    xrt::bo bo_b;
    xrt::bo bo_c;
    xrt::bo bo_tmp;
    xrt::bo bo_trace;

    // Tile dimensions baked into the compiled xclbin.
    int64_t tile_m = 32;
    int64_t tile_k = 32;
    int64_t tile_n = 32;

    // Minimum M*N*K to bother sending to the NPU.
    int64_t min_batch = 32768;

    // Per-tile NPU kernel timeout in milliseconds.
    // Minimum 1 (env_int rejects 0). Use INT64_MAX for effectively no timeout.
    // Default 5000 ms recommended for server deployments.
    int64_t timeout_ms = 5000;

    // Scratch buffers (resized on demand, host-visible).
    std::vector<int8_t>  quant_a;   // quantised weight matrix [M*K]
    std::vector<int8_t>  quant_b;   // quantised activation matrix [N*K]
    std::vector<float>   fp32_buf;  // temp F32 for dequant of src types
    std::vector<float>   scales_a;  // per-row quant scales for src0 (weights)
    std::vector<float>   scales_b;  // per-row quant scales for src1 (activations)

    // Tile scratch buffers (resized per-call in mul_mat, avoiding heap churn).
    std::vector<int8_t>  tile_a;
    std::vector<int8_t>  tile_b;
    std::vector<int32_t> tile_c;
    std::vector<int32_t> acc;

    // Serialises concurrent mul_mat calls across all ggml_backend_t handles that
    // share this singleton context. The NPU dispatches one kernel at a time anyway;
    // this mutex prevents the scratch-buffer data race when the ggml scheduler
    // issues concurrent graph_compute calls on multiple handles.
    std::mutex dispatch_mutex;

    // True when xclbin + instructions loaded successfully.
    bool kernel_ready = false;

    explicit ggml_backend_xdna_context() {
        auto env_int = [](const char * name, int64_t def, int64_t max_val) -> int64_t {
            const char * v = std::getenv(name);
            if (!v) { return def; }
            const int64_t val = std::atoll(v);
            if (val <= 0 || val > max_val) {
                GGML_LOG_WARN("ggml_xdna: %s=%s is out of range (1–%lld), "
                              "using default %lld\n",
                              name, v, (long long)max_val, (long long)def);
                return def;
            }
            return val;
        };
        tile_m     = env_int("GGML_XDNA_TILE_M",    32,    1024);
        tile_k     = env_int("GGML_XDNA_TILE_K",    32,    1024);
        tile_n     = env_int("GGML_XDNA_TILE_N",    32,    1024);
        min_batch  = env_int("GGML_XDNA_MIN_BATCH", 32768, INT64_MAX);
        timeout_ms = env_int("GGML_XDNA_TIMEOUT_MS", 5000, INT64_MAX);
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<uint32_t> load_instr_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open instruction file: " + path);
    }
    const auto size = f.tellg();
    if (size < 0) {
        throw std::runtime_error("instruction file: seek failed: " + path);
    }
    if (size > 4 * 1024 * 1024) {
        throw std::runtime_error("instruction file exceeds 4 MB limit: " + path);
    }
    if (size % 4 != 0) {
        throw std::runtime_error("instruction file size not a multiple of 4: " + path);
    }
    f.seekg(0);
    std::vector<uint32_t> data(size / 4);
    f.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

// ---------------------------------------------------------------------------
// Tile dispatch
// ---------------------------------------------------------------------------

// Dispatch one TILE_M × TILE_K × TILE_N matmul tile using pre-allocated BOs.
// a_tile: int8 [tile_m × tile_k] (row-major)
// b_tile: int8 [tile_k × tile_n] (row-major, transposed inside kernel)
// c_tile: int32 [tile_m × tile_n] output
// Returns true on success, false on NPU timeout or hardware fault.
// Callers must propagate failure up to graph_compute as GGML_STATUS_FAILED.
static bool dispatch_tile(ggml_backend_xdna_context * ctx,
                          const int8_t * a_tile,
                          const int8_t * b_tile,
                          int32_t      * c_tile) {
    auto & kern  = *ctx->matmul_kernel;
    auto & instr = ctx->bo_instr;

    const int64_t tm = ctx->tile_m;
    const int64_t tk = ctx->tile_k;
    const int64_t tn = ctx->tile_n;

    std::memcpy(ctx->bo_a.map<int8_t *>(), a_tile, (size_t)tm * tk * sizeof(int8_t));
    std::memcpy(ctx->bo_b.map<int8_t *>(), b_tile, (size_t)tk * tn * sizeof(int8_t));

    ctx->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    constexpr unsigned int opcode = 3;
    auto run = kern(opcode, instr,
                    static_cast<uint32_t>(ctx->instr_data.size()), // safe: load_instr_file enforces 4 MB / 4 = 1M word limit
                    ctx->bo_a, ctx->bo_b, ctx->bo_c, ctx->bo_tmp, ctx->bo_trace);
    // env_int enforces val > 0, so timeout_ms is always >= 1 after construction.
    const ert_cmd_state run_state = run.wait(std::chrono::milliseconds(ctx->timeout_ms));
    if (run_state != ERT_CMD_STATE_COMPLETED) {
        GGML_LOG_ERROR("ggml_xdna: NPU kernel dispatch failed (state=%d) — "
                       "returning GGML_STATUS_FAILED to caller\n",
                       static_cast<int>(run_state));
        return false;
    }

    ctx->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // kernel always fully overwrites bo_c (never accumulates); safe to read directly.
    std::memcpy(c_tile, ctx->bo_c.map<const int32_t *>(), (size_t)tm * tn * sizeof(int32_t));
    return true;
}

// ---------------------------------------------------------------------------
// Core matmul — tiled
// ---------------------------------------------------------------------------

// Returns false if any tile dispatch fails (NPU timeout/fault); caller returns GGML_STATUS_FAILED.
static bool ggml_backend_xdna_mul_mat(ggml_backend_xdna_context * ctx,
                                      struct ggml_tensor          * dst) {
    // Singleton context: serialise concurrent mul_mat calls across all handles
    // backed by this context (data race on scratch vectors).
    std::unique_lock<std::mutex> lk(ctx->dispatch_mutex);

    GGML_ASSERT(ctx->kernel_ready && "mul_mat called without a loaded kernel");
    const struct ggml_tensor * src0 = dst->src[0]; // weights     [K, M]
    const struct ggml_tensor * src1 = dst->src[1]; // activations [K, N]

    GGML_TENSOR_BINARY_OP_LOCALS

    // ggml GEMM: dst[M, N] = src1[N, K] * src0[M, K]^T
    const int64_t M = ne01; // weight rows
    const int64_t K = ne10; // inner dimension
    const int64_t N = ne11; // activation batch

    const int64_t tm = ctx->tile_m;
    const int64_t tk = ctx->tile_k;
    const int64_t tn = ctx->tile_n;

    // --- Dequantise all of src0 (weights) to F32 ---
    // Resize once to max(M*K, N*K) so the second phase never triggers a realloc.
    ctx->fp32_buf.resize(std::max(M * K, N * K));
    if (src0->type == GGML_TYPE_F32) {
        for (int64_t m = 0; m < M; m++) {
            std::memcpy(ctx->fp32_buf.data() + m * K,
                        (const char *) src0->data + m * nb01,
                        K * sizeof(float));
        }
    } else {
        const auto * tt = ggml_get_type_traits(src0->type);
        for (int64_t m = 0; m < M; m++) {
            tt->to_float((const char *) src0->data + m * nb01,
                         ctx->fp32_buf.data() + m * K, K);
        }
    }
    // Quantise weight matrix to int8 with per-row scales.
    ctx->quant_a.resize(M * K);
    ctx->scales_a.resize(M);
    quant_f32_to_int8(ctx->fp32_buf.data(), ctx->quant_a.data(), M, K, ctx->scales_a.data());

    // --- Dequantise all of src1 (activations) to F32 ---
    // fp32_buf is already sized to max(M*K, N*K) above; no realloc needed here.
    if (src1->type == GGML_TYPE_F32) {
        for (int64_t n = 0; n < N; n++) {
            std::memcpy(ctx->fp32_buf.data() + n * K,
                        (const char *) src1->data + n * nb11,
                        K * sizeof(float));
        }
    } else {
        const auto * tt = ggml_get_type_traits(src1->type);
        for (int64_t n = 0; n < N; n++) {
            tt->to_float((const char *) src1->data + n * nb11,
                         ctx->fp32_buf.data() + n * K, K);
        }
    }
    // Quantise activation matrix to int8 with per-row scales.
    ctx->quant_b.resize(N * K);
    ctx->scales_b.resize(N);
    quant_f32_to_int8(ctx->fp32_buf.data(), ctx->quant_b.data(), N, K, ctx->scales_b.data());

    // --- Output accumulator (float, zero-initialised) ---
    float * out = (float *) dst->data;
    std::fill(out, out + M * N, 0.0f);

    // --- Tile buffers (reuse context-resident vectors to avoid per-call heap churn) ---
    const int64_t tile_a_sz = tm * tk;
    const int64_t tile_b_sz = tk * tn;
    const int64_t tile_c_sz = tm * tn;

    ctx->tile_a.resize(tile_a_sz);
    ctx->tile_b.resize(tile_b_sz);
    ctx->tile_c.resize(tile_c_sz);

    // Per-(m0,n0) accumulator — accumulates NPU int32 tile results across K slices.
    // int32 is sufficient: worst case = tile_k * 127^2 (tile_k ≤ 1024 via env_int cap)
    // = 1024 * 16129 ≈ 16.5 M, well within INT32_MAX (2.1 B).
    ctx->acc.resize(tile_c_sz);

    // Tile over M, N, K with padding on edges.
    for (int64_t m0 = 0; m0 < M; m0 += tm) {
        for (int64_t n0 = 0; n0 < N; n0 += tn) {
            std::fill(ctx->acc.begin(), ctx->acc.end(), 0);

            for (int64_t k0 = 0; k0 < K; k0 += tk) {
                // Copy + zero-pad tile_a from quant_a[m0:m0+tm, k0:k0+tk]
                std::fill(ctx->tile_a.begin(), ctx->tile_a.end(), 0);
                for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                    const int64_t cols = std::min(tk, K - k0);
                    std::memcpy(ctx->tile_a.data() + mi * tk,
                                ctx->quant_a.data() + (m0 + mi) * K + k0,
                                cols); // cols * sizeof(int8_t) = cols; sizeof(int8_t)==1 by C++ std
                }

                // Copy + zero-pad tile_b from quant_b[n0:n0+tn, k0:k0+tk]
                // quant_b is [N, K]; kernel expects B in [K, N] layout for B^T.
                // Transpose during tile copy.
                std::fill(ctx->tile_b.begin(), ctx->tile_b.end(), 0);
                for (int64_t ni = 0; ni < tn && (n0 + ni) < N; ni++) {
                    for (int64_t ki = 0; ki < tk && (k0 + ki) < K; ki++) {
                        // tile_b[ki, ni] = quant_b[n0+ni, k0+ki]
                        ctx->tile_b[ki * tn + ni] =
                            ctx->quant_b[(n0 + ni) * K + (k0 + ki)];
                    }
                }

                if (!dispatch_tile(ctx, ctx->tile_a.data(), ctx->tile_b.data(), ctx->tile_c.data())) {
                    return false;
                }

                for (int64_t i = 0; i < tile_c_sz; i++) {
                    ctx->acc[i] += ctx->tile_c[i];
                }
            }

            // Scatter accumulated result into output with per-element dequantisation.
            // dst is zero-filled above; use = not += to be safe if zero-fill is ever removed.
            for (int64_t mi = 0; mi < tm && (m0 + mi) < M; mi++) {
                for (int64_t ni = 0; ni < tn && (n0 + ni) < N; ni++) {
                    out[(m0 + mi) * N + (n0 + ni)] =
                        static_cast<float>(ctx->acc[mi * tn + ni])
                        * ctx->scales_a[m0 + mi] * ctx->scales_b[n0 + ni];
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

// Process-lifetime singleton — allocated once in ggml_backend_xdna_reg(), never freed.
// Shared by all backends returned by ggml_backend_xdna_init() to avoid opening
// a second xrt::hw_context on the same device.
static ggml_backend_xdna_context * g_probe_ctx = nullptr;

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    // g_probe_ctx is a process-lifetime singleton (never freed); only delete
    // contexts that were independently allocated.
    auto * ctx = (ggml_backend_xdna_context *) backend->context;
    if (ctx != g_probe_ctx) {
        delete ctx;
    }
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t     backend,
                                                         struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_xdna_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { continue; }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_xdna_mul_mat(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("ggml_xdna: graph_compute received op %s — "
                           "supports_op should have prevented this\n",
                           ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i xdna_backend_i = {
    /* .get_name            = */ ggml_backend_xdna_get_name,
    /* .free                = */ ggml_backend_xdna_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_xdna_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = {
        0xa3, 0x7d, 0x4e, 0x21, 0xb8, 0x55, 0x4c, 0x09,
        0x9e, 0x12, 0xf0, 0x3a, 0x77, 0xc6, 0x81, 0xd4
    };
    return &guid;
}

// ---------------------------------------------------------------------------
// Device interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "XDNA";
}
static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AMD XDNA2 NPU (RyzenAI)";
}
static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0; *total = 0;
}
static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev,
                                                struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /*async_tensor=*/false, /*events=*/false,
                    /*host_buffer=*/true, /*buffer_from_host_ptr=*/false };
}
static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev); GGML_UNUSED(params);
    return ggml_backend_xdna_init();
}
static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}
static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t       dev,
                                                   const struct ggml_tensor * op) {
    auto * ctx = (ggml_backend_xdna_context *) dev->context;

    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            if (!ctx || !ctx->kernel_ready) { return false; }

            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            if (!ggml_is_contiguous(src0)) { return false; }
            if (!ggml_is_contiguous(src1)) { return false; }
            if (src1->type != GGML_TYPE_F32) { return false; } // BF16/F16 activations not yet supported
            if (src0->type != GGML_TYPE_F32 &&
                ggml_get_type_traits(src0->type)->to_float == NULL) { return false; }
            if (op->ne[2] != 1 || op->ne[3] != 1) { return false; }

            const int64_t M = src0->ne[1];
            const int64_t K = src1->ne[0];
            const int64_t N = src1->ne[1];
            // Guard int32 accumulator overflow: with tile_k ≤ 1024 (env_int cap),
            // each tile contributes at most tile_k * 127^2 ≈ 16.5M; summing
            // ceil(K/tile_k) tiles overflows INT32_MAX when K > ~133K.
            if (K > 131072) { return false; }
            if ((double)M * N * K < (double)ctx->min_batch) { return false; }

            return true;
        }
        default: return false;
    }
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev,
                                                     ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// Registry + init
// ---------------------------------------------------------------------------

// g_probe_ctx — see definition above (before backend interface section).

static bool try_init_context(ggml_backend_xdna_context * ctx) {
    const char * xclbin_path = std::getenv("GGML_XDNA_XCLBIN_PATH");
    const char * instr_path  = std::getenv("GGML_XDNA_INSTR_PATH");

    if (!xclbin_path || !instr_path) {
        GGML_LOG_INFO("ggml_xdna: GGML_XDNA_XCLBIN_PATH or GGML_XDNA_INSTR_PATH "
                      "not set — kernel dispatch disabled\n");
        return false;
    }

    // Canonicalise paths before passing to XRT/ifstream to catch traversal
    // attempts and produce a clear error before the XRT mmap error fires.
    auto resolve_path = [](const char * raw, const char * var) -> std::string {
        char resolved[4096];
        if (!realpath(raw, resolved)) {
            throw std::runtime_error(std::string(var) + ": cannot resolve path");
        }
        return std::string(resolved);
    };
    const std::string xclbin_str = resolve_path(xclbin_path, "GGML_XDNA_XCLBIN_PATH");
    const std::string instr_str  = resolve_path(instr_path,  "GGML_XDNA_INSTR_PATH");

    ctx->device = xrt::device(0);

    xrt::xclbin bin(xclbin_str);
    ctx->device.register_xclbin(bin);

    ctx->hw_ctx = std::make_unique<xrt::hw_context>(ctx->device, bin.get_uuid());
    ctx->matmul_kernel = std::make_unique<xrt::kernel>(*ctx->hw_ctx, "MLIR_AIE");

    ctx->instr_data = load_instr_file(instr_str);
    ctx->bo_instr   = xrt::bo(ctx->device,
                               ctx->instr_data.size() * sizeof(uint32_t),
                               xrt::bo::flags::cacheable,
                               ctx->matmul_kernel->group_id(1));
    std::memcpy(ctx->bo_instr.map<uint32_t *>(),
                ctx->instr_data.data(),
                ctx->instr_data.size() * sizeof(uint32_t));
    ctx->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Pre-allocate data BOs sized for one tile each.
    auto & kern     = *ctx->matmul_kernel;
    ctx->bo_a       = xrt::bo(ctx->device,
                               (size_t)ctx->tile_m * ctx->tile_k * sizeof(int8_t),
                               xrt::bo::flags::host_only, kern.group_id(3));
    ctx->bo_b       = xrt::bo(ctx->device,
                               (size_t)ctx->tile_k * ctx->tile_n * sizeof(int8_t),
                               xrt::bo::flags::host_only, kern.group_id(4));
    ctx->bo_c       = xrt::bo(ctx->device,
                               (size_t)ctx->tile_m * ctx->tile_n * sizeof(int32_t),
                               xrt::bo::flags::host_only, kern.group_id(5));
    ctx->bo_tmp     = xrt::bo(ctx->device, 4,
                               xrt::bo::flags::host_only, kern.group_id(6));
    ctx->bo_trace   = xrt::bo(ctx->device, 4,
                               xrt::bo::flags::host_only, kern.group_id(7));

    ctx->kernel_ready = true;
    GGML_LOG_INFO("ggml_xdna: kernel ready — tile %lld×%lld×%lld\n",
                  (long long)ctx->tile_m, (long long)ctx->tile_k,
                  (long long)ctx->tile_n);
    return true;
}

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}
static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}
static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { xdna_device_i, reg, nullptr };
    static std::once_flag dev_ctx_once;
    std::call_once(dev_ctx_once, [&]() {
        GGML_ASSERT(g_probe_ctx && "dev_ctx_once: g_probe_ctx is null — "
                    "ggml_backend_xdna_reg() must be called before reg_get_device()");
        dev.context = g_probe_ctx;
    });
    return &dev;
}
static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg); GGML_UNUSED(name);
    return NULL;
}
static const struct ggml_backend_reg_i xdna_reg_i = {
    ggml_backend_xdna_reg_get_name,
    ggml_backend_xdna_reg_get_device_count,
    ggml_backend_xdna_reg_get_device,
    ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, xdna_reg_i, NULL };

    static std::once_flag g_probe_once;
    std::call_once(g_probe_once, []() {
        auto * probe = new ggml_backend_xdna_context();
        try {
            // g_probe_ctx is reused by ggml_backend_xdna_init(); only one hw_context
            // is ever opened on this device — no XDNA2 firmware arbitration conflict.
            try_init_context(probe);
        } catch (const std::exception & e) {
            GGML_LOG_WARN("ggml_xdna: probe failed: %s\n", e.what());
        }
        g_probe_ctx = probe;
    });
    return &reg;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_reg(); // ensure probe ran; g_probe_ctx is valid after this

    // Reuse the probe context directly — avoids opening a second xrt::hw_context
    // on the same device, which can cause firmware arbitration failures on XDNA2.
    if (!g_probe_ctx) {
        GGML_LOG_WARN("ggml_xdna: init called but probe context is null\n");
        return NULL;
    }

    return new ggml_backend {
        ggml_backend_xdna_guid(),
        xdna_backend_i,
        ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        g_probe_ctx,
    };
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
