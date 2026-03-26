/*
 * int4_quant_sanity.c — quantization error comparison: int8 vs int4
 *
 * Dequantizes weight tensors from a Q4_K_M GGUF model, then applies
 * uniform symmetric int8 and int4 per-block requantization (matching
 * ggml-xdna.cpp) and reports RMSE / SNR.
 *
 * Build:
 *   gcc -O2 -o /tmp/int4_sanity tools/int4_quant_sanity.c \
 *       -I./ggml/include -I./ggml/src -I./include \
 *       -L./build/ggml/src -lggml-base \
 *       -Wl,-rpath,./build/ggml/src -lm
 */

#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define BLOCK_SIZE 32

static double rmse_int8(const float *x, int64_t n) {
    double sse = 0.0;
    for (int64_t i = 0; i < n; ) {
        int blk = (n - i < BLOCK_SIZE) ? (int)(n - i) : BLOCK_SIZE;
        float amax = 0.f;
        for (int j = 0; j < blk; j++) { float v = fabsf(x[i+j]); if (v > amax) amax = v; }
        float sc = (amax > 0.f) ? (amax / 127.f) : 1.f;
        for (int j = 0; j < blk; j++) {
            int q = (int)roundf(x[i+j] / sc);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            double e = x[i+j] - (float)q * sc;
            sse += e * e;
        }
        i += blk;
    }
    return sqrt(sse / (double)n);
}

static double rmse_int4(const float *x, int64_t n) {
    double sse = 0.0;
    for (int64_t i = 0; i < n; ) {
        int blk = (n - i < BLOCK_SIZE) ? (int)(n - i) : BLOCK_SIZE;
        float amax = 0.f;
        for (int j = 0; j < blk; j++) { float v = fabsf(x[i+j]); if (v > amax) amax = v; }
        float sc = (amax > 0.f) ? (amax / 7.f) : 1.f;
        for (int j = 0; j < blk; j++) {
            int q = (int)roundf(x[i+j] / sc);
            if (q >  7) q =  7;
            if (q < -8) q = -8;
            double e = x[i+j] - (float)q * sc;
            sse += e * e;
        }
        i += blk;
    }
    return sqrt(sse / (double)n);
}

static const char *TARGETS[] = {
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",
    "blk.15.attn_q.weight",
    "blk.15.ffn_gate.weight",
    "blk.31.attn_q.weight",
    "blk.31.ffn_down.weight",
    NULL
};

int main(int argc, char **argv) {
    const char *model_path = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    if (argc > 1) model_path = argv[1];

    struct ggml_context *meta = NULL;
    struct gguf_init_params ip = { .no_alloc = true, .ctx = &meta };
    struct gguf_context *gc = gguf_init_from_file(model_path, ip);
    if (!gc) { fprintf(stderr, "Failed to open %s\n", model_path); return 1; }

    printf("Model: %s\n", model_path);
    printf("\n%-42s %-8s %12s %12s %8s %8s %6s\n",
           "Tensor", "type", "int8_RMSE", "int4_RMSE", "SNR_i8", "SNR_i4", "ratio");
    printf("%.96s\n", "------------------------------------------------------------"
                      "------------------------------------------------------------");

    double sr8=0, sr4=0, ss8=0, ss4=0; int cnt=0;

    for (int ti = 0; TARGETS[ti]; ti++) {
        const char *name = TARGETS[ti];
        int idx = gguf_find_tensor(gc, name);
        if (idx < 0) { printf("  %-40s NOT FOUND\n", name); continue; }

        struct ggml_tensor *t = ggml_get_tensor(meta, name);
        if (!t) { printf("  %-40s ggml_get_tensor failed\n", name); continue; }

        int64_t ne = ggml_nelements(t);
        float *f32 = malloc(ne * sizeof(float));
        void  *raw = malloc(ggml_nbytes(t));
        if (!f32 || !raw) { fprintf(stderr, "OOM\n"); return 1; }

        FILE *fp = fopen(model_path, "rb");
        fseek(fp, (long)(gguf_get_data_offset(gc) + gguf_get_tensor_offset(gc, idx)), SEEK_SET);
        fread(raw, 1, ggml_nbytes(t), fp);
        fclose(fp);

        const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            printf("  %-40s no dequantizer\n", name);
            free(f32); free(raw); continue;
        }
        tr->to_float(raw, f32, ne);
        free(raw);

        /* signal power */
        double sig = 0.0;
        for (int64_t i = 0; i < ne; i++) sig += (double)f32[i]*f32[i];
        sig /= ne;

        double r8 = rmse_int8(f32, ne);
        double r4 = rmse_int4(f32, ne);
        double s8 = (r8 > 0) ? 10.0*log10(sig/(r8*r8)) : 99.0;
        double s4 = (r4 > 0) ? 10.0*log10(sig/(r4*r4)) : 99.0;
        double ratio = (r8 > 0) ? r4/r8 : 0.0;

        printf("  %-40s %-8s %12.6f %12.6f %8.1f %8.1f %6.1fx\n",
               name, ggml_type_name(t->type), r8, r4, s8, s4, ratio);

        sr8+=r8; sr4+=r4; ss8+=s8; ss4+=s4; cnt++;
        free(f32);
    }

    if (cnt) {
        printf("%.96s\n", "------------------------------------------------------------"
                          "------------------------------------------------------------");
        printf("  %-40s %-8s %12.6f %12.6f %8.1f %8.1f %6.1fx\n",
               "AVERAGE", "", sr8/cnt, sr4/cnt, ss8/cnt, ss4/cnt, (sr4/cnt)/(sr8/cnt));
    }

    printf("\nNote:\n");
    printf("  Current path:  Q4_K_M -> f32 -> int8 -> NPU   (SNR_i8 column)\n");
    printf("  Proposed path: Q4_K_M -> f32 -> int4 -> NPU   (SNR_i4 column)\n");
    printf("  SNR > 30 dB = generally acceptable for LLM inference\n");
    printf("  ratio = how many times worse int4 RMSE is vs int8\n");

    gguf_free(gc);
    if (meta) ggml_free(meta);
    return 0;
}
