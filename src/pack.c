#include "gemm.h"
#include <arm_neon.h>

// Common 4x4 NEON transpose: the packing implementation is identical for all
// three hand-written GEMMs. Only the arithmetic kernels change.
void pack_inputs(size_t n, const float *restrict a, const float *restrict b,
                 float *restrict ap, float *restrict bp) {
    for (size_t pc = 0; pc < n; pc += KC) {
        const size_t kc = n - pc < KC ? n - pc : KC;
        for (size_t panel = 0; panel < n; panel += PANEL) {
            float *pa = ap + pc * n + panel * kc;
            float *pb = bp + pc * n + panel * kc;
            for (size_t k = 0; k < kc; k += 4) {
                for (size_t r = 0; r < PANEL; r += 4) {
                    const float32x4_t r0 = vld1q_f32(a + (panel+r+0)*n + pc+k);
                    const float32x4_t r1 = vld1q_f32(a + (panel+r+1)*n + pc+k);
                    const float32x4_t r2 = vld1q_f32(a + (panel+r+2)*n + pc+k);
                    const float32x4_t r3 = vld1q_f32(a + (panel+r+3)*n + pc+k);
                    const float32x4x2_t t01 = vtrnq_f32(r0, r1);
                    const float32x4x2_t t23 = vtrnq_f32(r2, r3);
                    vst1q_f32(pa+(k+0)*PANEL+r, vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0])));
                    vst1q_f32(pa+(k+1)*PANEL+r, vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1])));
                    vst1q_f32(pa+(k+2)*PANEL+r, vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0])));
                    vst1q_f32(pa+(k+3)*PANEL+r, vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1])));
                }
            }
            for (size_t k = 0; k < kc; ++k) {
                const float *source = b + (pc+k)*n + panel;
                float *target = pb + k*PANEL;
                vst1q_f32(target+0, vld1q_f32(source+0));
                vst1q_f32(target+4, vld1q_f32(source+4));
                vst1q_f32(target+8, vld1q_f32(source+8));
                vst1q_f32(target+12, vld1q_f32(source+12));
            }
        }
    }
}
