#include "gemm.h"
#include <arm_sme.h>
#include <stdint.h>

// Transpose 16x16 blocks in ZA instead of dozens of NEON shuffle instructions.
__arm_locally_streaming __arm_new("za")
void pack_a_opt_sme(size_t n, const float *restrict a, float *restrict ap) {
    const svbool_t pg = svptrue_b32();
    for (size_t pc = 0; pc < n; pc += OPT_KC) {
        const size_t kc = n-pc < OPT_KC ? n-pc : OPT_KC;
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
