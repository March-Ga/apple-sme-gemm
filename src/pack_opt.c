#include "gemm.h"
#include <arm_neon.h>

// A: panels 16 x kc, column-major within each panel.
// B: panels kc x 64, row-major within each panel (four contiguous Z vectors).
// Blocks start at pc*n; unlike the controlled baseline, this layout is tuned
// jointly with the SME2 kernel. No padding, extra allocation or hidden BLAS.
void pack_inputs_opt(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict ap, float *restrict bp) {
    if (n <= DIRECT_B_MAX) { pack_a_opt_sme(n, a, ap); return; }
    if (n%64) { pack_inputs(n, a, b, ap, bp); return; }
    pack_a_opt_sme(n, a, ap);
    for (size_t pc = 0; pc < n; pc += OPT_KC) {
        const size_t kc = n-pc < OPT_KC ? n-pc : OPT_KC;
        for (size_t jc = 0; jc < n; jc += 256) {
            const size_t je = n-jc < 256 ? n : jc+256;
            for (size_t kk = 0; kk < kc; kk += 32) {
                const size_t ke = kc-kk < 32 ? kc : kk+32;
                for (size_t k = kk; k < ke; ++k)
                    for (size_t j = jc; j < je; j += 64) {
                        const float *src = b+(pc+k)*n+j;
                        float *dst = bp+pc*n+j*kc+k*64;
                        for (size_t v = 0; v < 64; v += 16)
                            vst1q_f32_x4(dst+v, vld1q_f32_x4(src+v));
                    }
            }
        }
    }
}
