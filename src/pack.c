#include "gemm.h"
#include <arm_sme.h>
#include <stdint.h>
#include <arm_neon.h>

// Transpose 16x16 blocks in ZA instead of dozens of NEON shuffle instructions.
__arm_locally_streaming __arm_new("za")
static void pack_a(size_t n, const float *restrict a, float *restrict ap) {
    const svbool_t pg = svptrue_b32();
    for (size_t pc = 0; pc < n; pc += KC) {
        const size_t kc = n-pc < KC ? n-pc : KC;
        for (size_t i = 0; i < n; i += 16)
            for (size_t k = 0; k < kc; k += 16) {
                #pragma clang loop unroll(disable)
                for (uint32_t row = 0; row < 16; ++row)
                    svld1_hor_za32(0, row, pg, a+(i+row)*n+pc+k);
                #pragma clang loop unroll(disable)
                for (uint32_t col = 0; col < 16; col += 4) {
                    const svfloat32x4_t columns = svread_ver_za32_f32_vg4(0, col);
                    svst1_f32_x4(svptrue_c32(), ap+pc*n+i*kc+(k+col)*16, columns);
                }
            }
    }
}

// A: 16-row column-major panels. B: 64-column row-major panels.
// Only a final 32-column B tail is zero-padded; C retains its original stride.
void pack_inputs(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict ap, float *restrict bp) {
    if (n <= DIRECT_B_MAX) { pack_a(n, a, ap); return; }
    pack_a(n, a, ap);
    const size_t full_columns = n/64*64;
    const size_t b_columns = (n+63)/64*64;
    for (size_t pc = 0; pc < n; pc += KC) {
        const size_t kc = n-pc < KC ? n-pc : KC;
        for (size_t jc = 0; jc < full_columns; jc += 256) {
            const size_t je = full_columns-jc < 256 ? full_columns : jc+256;
            for (size_t kk = 0; kk < kc; kk += 32) {
                const size_t ke = kc-kk < 32 ? kc : kk+32;
                for (size_t k = kk; k < ke; ++k)
                    for (size_t j = jc; j < je; j += 64) {
                        const float *src = b+(pc+k)*n+j;
                        float *dst = bp+pc*b_columns+j*kc+k*64;
                        for (size_t v = 0; v < 64; v += 16)
                            vst1q_f32_x4(dst+v, vld1q_f32_x4(src+v));
                    }
            }
        }
        if (full_columns < n) {
            for (size_t k = 0; k < kc; ++k) {
                const float *src = b+(pc+k)*n+full_columns;
                float *dst = bp+pc*b_columns+full_columns*kc+k*64;
                for (size_t v = 0; v < 32; v += 16)
                    vst1q_f32_x4(dst+v, vld1q_f32_x4(src+v));
                for (size_t v = 32; v < 64; v += 4)
                    vst1q_f32(dst+v, vdupq_n_f32(0));
            }
        }
    }
}
