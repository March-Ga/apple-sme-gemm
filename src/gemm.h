#ifndef APPLE_SME_GEMM_H
#define APPLE_SME_GEMM_H
#include <stddef.h>

enum { PANEL = 16, KC = 2048, MC = 256, NC = 512, DIRECT_B_MAX = 512 };

// Square row-major FP32 inputs; N is a multiple of 32 in [32,8192].
// A workspace: N*N floats. B workspace: sme_packed_b_count(N) floats.
// Small N uses original B directly; larger N uses packed A16/B64 panels.
static inline size_t sme_packed_b_count(size_t n) {
    return n <= DIRECT_B_MAX ? 0 : n*((n+63)/64*64);
}
void pack_inputs(size_t n, const float *restrict a, const float *restrict b,
                 float *restrict ap, float *restrict bp);
void gemm_sme(size_t n, const float *restrict ap, const float *restrict b,
              float *restrict c);
size_t sme_vector_lanes(void);
int accelerate_prepare(void);
void gemm_accelerate(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict c);
#endif
