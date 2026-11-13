#include "gemm.h"
#include <arm_sme.h>
#include <stdint.h>

// Small matrices fit in cache: keep B row-major and avoid its packing entirely.
__arm_locally_streaming __arm_new("za")
static void direct_b(size_t n, const float *restrict ap, const float *restrict b,
                      float *restrict c) {
    const svbool_t pg = svptrue_b32();
    const svcount_t pg2 = svptrue_c32();
    for (size_t i = 0; i < n; i += 32)
        for (size_t j = 0; j < n; j += 32) {
            svzero_za();
            for (size_t k = 0; k < n; ++k) {
                const svfloat32_t a0 = svld1_f32(pg, ap+i*n+k*16);
                const svfloat32_t a1 = svld1_f32(pg, ap+(i+16)*n+k*16);
                const svfloat32x2_t bv = svld1_f32_x2(pg2, b+k*n+j);
                const svfloat32_t b0 = svget2_f32(bv, 0), b1 = svget2_f32(bv, 1);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);
            }
            #pragma clang loop unroll(disable)
            for (uint32_t row = 0; row < 16; ++row) {
                svst1_hor_za32(0, row, pg, c+(i+row)*n+j);
                svst1_hor_za32(1, row, pg, c+(i+row)*n+j+16);
                svst1_hor_za32(2, row, pg, c+(i+row+16)*n+j);
                svst1_hor_za32(3, row, pg, c+(i+row+16)*n+j+16);
            }
        }
}

static inline __attribute__((always_inline))
void tile_16x64(const float *restrict a, const float *restrict b,
                float *restrict c, size_t n, size_t kc, int accumulate, int full_width)
    __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();
    if (accumulate) {
        #pragma clang loop unroll(disable)
        for (uint32_t row = 0; row < 16; ++row) {
            svld1_hor_za32(0, row, pg, c+row*n);
            svld1_hor_za32(1, row, pg, c+row*n+16);
            if (full_width) {
                svld1_hor_za32(2, row, pg, c+row*n+32);
                svld1_hor_za32(3, row, pg, c+row*n+48);
            }
        }
    } else svzero_za();

    // Consume each tuple in this iteration to keep streaming-register live ranges short.
    #pragma clang loop unroll_count(4)
    for (size_t k = 0; k < kc; ++k) {
        const svfloat32_t va = svld1_f32(pg, a+k*16);
        const svfloat32x4_t vb = svld1_f32_x4(pg4, b+k*64);
        svmopa_za32_f32_m(0, pg, pg, va, svget4_f32(vb, 0));
        svmopa_za32_f32_m(1, pg, pg, va, svget4_f32(vb, 1));
        svmopa_za32_f32_m(2, pg, pg, va, svget4_f32(vb, 2));
        svmopa_za32_f32_m(3, pg, pg, va, svget4_f32(vb, 3));
    }
    #pragma clang loop unroll(disable)
    for (uint32_t row = 0; row < 16; ++row) {
        svst1_hor_za32(0, row, pg, c+row*n);
        svst1_hor_za32(1, row, pg, c+row*n+16);
        if (full_width) {
            svst1_hor_za32(2, row, pg, c+row*n+32);
            svst1_hor_za32(3, row, pg, c+row*n+48);
        }
    }
}

// Packed B uses 64 columns per panel; the final panel may contain 32 zero columns.
__arm_locally_streaming __arm_new("za")
static void optimized(size_t n, const float *restrict ap, const float *restrict bp,
                       float *restrict c) {
    const size_t b_columns = (n+63)/64*64;
    for (size_t pc = 0; pc < n; pc += KC) {
        const size_t kc = n-pc < KC ? n-pc : KC;
        for (size_t jc = 0; jc < n; jc += NC) {
            const size_t je = n-jc < NC ? n : jc+NC;
            for (size_t ic = 0; ic < n; ic += MC) {
                const size_t ie = n-ic < MC ? n : ic+MC;
                for (size_t j = jc; j < je; j += 64)
                    for (size_t i = ic; i < ie; i += 16)
                        tile_16x64(ap+pc*n+i*kc, bp+pc*b_columns+j*kc,
                                    c+i*n+j, n, kc, pc != 0, n-j >= 64);
            }
        }
    }
}

void gemm_sme(size_t n, const float *restrict ap, const float *restrict bp,
                  float *restrict c) {
    if (n <= DIRECT_B_MAX) direct_b(n, ap, bp, c);
    else optimized(n, ap, bp, c);
}

size_t sme_vector_lanes(void) { return svcntsw(); }
