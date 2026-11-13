#include "gemm.h"
#include <arm_sme.h>
#include <stdint.h>

static inline __attribute__((always_inline))
void matmul_32x32_sme(const float *restrict a0, const float *restrict a1,
                      const float *restrict b0, const float *restrict b1,
                      float *restrict c, size_t ldc, size_t k_size, int accumulate)
    __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    if (accumulate) {
        for (uint32_t row = 0; row < PANEL; ++row) {
            svld1_hor_za32(0, row, pg, c+row*ldc);
            svld1_hor_za32(1, row, pg, c+row*ldc+PANEL);
            svld1_hor_za32(2, row, pg, c+(row+PANEL)*ldc);
            svld1_hor_za32(3, row, pg, c+(row+PANEL)*ldc+PANEL);
        }
    } else {
        svzero_za();
    }
    for (size_t k = 0; k < k_size; ++k) {
        const svfloat32_t va0 = svld1_f32(pg, a0+k*PANEL);
        const svfloat32_t va1 = svld1_f32(pg, a1+k*PANEL);
        const svfloat32_t vb0 = svld1_f32(pg, b0+k*PANEL);
        const svfloat32_t vb1 = svld1_f32(pg, b1+k*PANEL);
        svmopa_za32_f32_m(0, pg, pg, va0, vb0);
        svmopa_za32_f32_m(1, pg, pg, va0, vb1);
        svmopa_za32_f32_m(2, pg, pg, va1, vb0);
        svmopa_za32_f32_m(3, pg, pg, va1, vb1);
    }
    for (uint32_t row = 0; row < PANEL; ++row) {
        svst1_hor_za32(0, row, pg, c+row*ldc);
        svst1_hor_za32(1, row, pg, c+row*ldc+PANEL);
        svst1_hor_za32(2, row, pg, c+(row+PANEL)*ldc);
        svst1_hor_za32(3, row, pg, c+(row+PANEL)*ldc+PANEL);
    }
}

__arm_locally_streaming __arm_new("za")
void gemm_sme32(size_t n, const float *restrict ap, const float *restrict bp,
                float *restrict c) {
    for (size_t pc = 0; pc < n; pc += KC) {
        const size_t kc = n-pc < KC ? n-pc : KC;
        for (size_t i = 0; i < n; i += 2*PANEL)
            for (size_t j = 0; j < n; j += 2*PANEL)
                matmul_32x32_sme(ap+pc*n+i*kc, ap+pc*n+(i+PANEL)*kc,
                                 bp+pc*n+j*kc, bp+pc*n+(j+PANEL)*kc,
                                 c+i*n+j, n, kc, pc != 0);
    }
}
