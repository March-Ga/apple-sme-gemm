// Packing correctness only: does not call a GEMM or measure performance.
#include "../src/gemm.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check(size_t n) {
    const size_t count = n*n;
    float *a = malloc(count*sizeof(float)), *b = malloc(count*sizeof(float));
    float *ap = malloc(count*sizeof(float)), *bp = malloc(count*sizeof(float));
    assert(a && b && ap && bp);
    for (size_t x = 0; x < count; ++x) {
        a[x] = (float)(x+1); b[x] = -(float)(x+1);
        ap[x] = NAN; bp[x] = NAN;
    }
    pack_inputs_opt(n, a, b, ap, bp);
    const size_t block = n <= DIRECT_B_MAX || n%64 == 0 ? OPT_KC : KC;
    const size_t width = n%64 ? PANEL : 64;
    // Reference walks original matrices and derives each packed destination.
    // Every destination must have been written, including the last K block.
    for (size_t row = 0; row < n; ++row)
        for (size_t col = 0; col < n; ++col) {
            const size_t pc_a = (col/block)*block;
            const size_t kc_a = n-pc_a < block ? n-pc_a : block;
            const size_t target_a = pc_a*n+(row/16)*16*kc_a+(col-pc_a)*16+row%16;
            assert(ap[target_a] == a[row*n+col]);
            if (n <= DIRECT_B_MAX) {
                assert(isnan(bp[row*n+col]));
                continue;
            }
            const size_t pc_b = (row/block)*block;
            const size_t kc_b = n-pc_b < block ? n-pc_b : block;
            const size_t target_b = pc_b*n+(col/width)*width*kc_b+(row-pc_b)*width+col%width;
            assert(bp[target_b] == b[row*n+col]);
        }
    free(a); free(b); free(ap); free(bp);
}

int main(void) {
    const size_t cases[] = {32, 64, 96, 128, 192, 288, 320, 544, 576, 1024, 1088, 2112};
    for (size_t x = 0; x < sizeof(cases)/sizeof(cases[0]); ++x) check(cases[x]);
    puts("Packing layout PASS (12 sizes); no GEMM executed.");
    return 0;
}
