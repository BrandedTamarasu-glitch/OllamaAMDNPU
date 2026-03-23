#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

/** Initialise the XDNA NPU backend.
 *
 * Uses XRT to open the first available NPU device and load a compiled xclbin
 * plus instruction sequence. Paths are read from the environment at startup:
 *   GGML_XDNA_XCLBIN_PATH — compiled xclbin        (required to dispatch)
 *   GGML_XDNA_INSTR_PATH  — _sequence.bin           (required to dispatch)
 *   GGML_XDNA_TILE_M      — tile rows    (default 32, range 1–1024)
 *   GGML_XDNA_TILE_K      — tile inner   (default 32, range 1–1024)
 *   GGML_XDNA_TILE_N      — tile cols    (default 32, range 1–1024)
 *   GGML_XDNA_MIN_BATCH   — minimum M×N×K to offload (default 32768)
 *   GGML_XDNA_TIMEOUT_MS  — per-tile NPU timeout in ms  (default 5000; range 1–INT64_MAX)
 *
 * Returns NULL if XRT initialisation fails. When the env vars are absent a
 * valid backend is still returned, but supports_op() returns false for all
 * ops so all computation falls back to CPU transparently.
 */
GGML_BACKEND_API ggml_backend_t     ggml_backend_xdna_init(void);

GGML_BACKEND_API bool               ggml_backend_is_xdna(ggml_backend_t backend);

/** Returns the XDNA backend registry entry.
 *  Preferred entry point for the ggml backend registry; call this rather
 *  than ggml_backend_xdna_init() unless you need a standalone instance.
 */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef  __cplusplus
}
#endif
