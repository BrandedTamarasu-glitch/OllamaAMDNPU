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
 *   GGML_XDNA_XCLBIN_PATH     — compiled xclbin        (required to dispatch)
 *   GGML_XDNA_INSTR_PATH      — _sequence.bin           (required to dispatch)
 *   GGML_XDNA_TILE_M          — tile rows    (default 32, range 1–131072)
 *   GGML_XDNA_TILE_K          — tile inner   (default 32, range 1–131072)
 *   GGML_XDNA_TILE_N          — tile cols    (default 32, range 1–131072)
 *   GGML_XDNA_MIN_BATCH       — minimum M×N×K to offload (default 32768)
 *   GGML_XDNA_MIN_N           — min activation batch to offload (default 2; set 1 for decode)
 *   GGML_XDNA_MAX_N           — max activation batch to offload (default 131072; set 1 for decode-only)
 *   GGML_XDNA_TIMEOUT_MS      — per-tile NPU timeout in ms  (default 5000; range 1–INT64_MAX)
 *
 * Up to 8 xclbin slots can be loaded for different K (or tile_n) dimensions.
 * Slots 2–8 use env suffix _2 through _8 (e.g. GGML_XDNA_XCLBIN_PATH_2).
 * See ggml-xdna.cpp header comment for full multi-slot documentation.
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
