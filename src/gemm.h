#ifndef INTRINSICS_GEMM_H
#define INTRINSICS_GEMM_H
#include <stddef.h>

enum { PANEL = 16, KC = 256 };
enum { OPT_KC = 2048, OPT_MC = 256, OPT_NC = 512 };
enum { DIRECT_B_MAX = 512 };
void pack_inputs_opt(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict ap, float *restrict bp);
void pack_a_opt_sme(size_t n, const float *restrict a, float *restrict ap);
void gemm_sme_opt(size_t n, const float *restrict ap, const float *restrict bp,
                  float *restrict c);

// A, B and C are square row-major matrices. N must be a multiple of 32.
// A/B packed layout, used by the original SME32 kernel:
// for each K block pc, kc=min(KC,N-pc), and each 16-lane panel starting at p:
//   panel = packed + pc*N + p*kc
//   panel[k*16 + lane] = A[(p+lane)*N + pc+k]  (A)
//   panel[k*16 + lane] = B[(pc+k)*N + p+lane]  (B)
void pack_inputs(size_t n, const float *restrict a, const float *restrict b,
                 float *restrict ap, float *restrict bp);
void gemm_sme32(size_t n, const float *restrict ap, const float *restrict bp,
                float *restrict c);
size_t sme_vector_lanes(void);
int accelerate_prepare(void);
void gemm_accelerate(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict c);
#endif
