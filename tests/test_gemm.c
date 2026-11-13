// Independent numerical regression tests, not a timing benchmark.
#include "../src/gemm.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t state;
static float value(void) {
    state = state*1664525u+1013904223u;
    return (float)((int)(state>>12)-524288)*(1.0f/524288.0f);
}
static float *memory(size_t count) {
    void *p = NULL;
    assert(posix_memalign(&p, 64, count*sizeof(float)) == 0);
    return p;
}
static void check(size_t n, unsigned seed) {
    const size_t count = n*n;
    float *a=memory(count), *b=memory(count), *ap=memory(count), *bp=memory(count);
    float *guarded=memory(count+32), *c=guarded+16;
    state=seed;
    for (size_t i=0;i<count;++i) { a[i]=value(); b[i]=value(); }
    for (size_t i=0;i<16;++i) guarded[i]=guarded[count+16+i]=12345;
    pack_inputs_opt(n,a,b,ap,bp);
    const float *right=n <= DIRECT_B_MAX ? b : bp;
    // Two products into a dirty buffer check that beta=0 is respected.
    for (unsigned repeat=0;repeat<2;++repeat) {
        for (size_t x=0;x<count;++x) c[x]=repeat ? 17 : NAN;
        gemm_sme_opt(n,ap,right,c);
        const size_t checks=n<=192 ? count : 512;
        for (size_t t=0;t<checks;++t) {
            const size_t position=n<=192 ? t : (t*104729+17)%count;
            const size_t row=position/n, col=position%n;
            double expected=0, scale=0;
            for (size_t k=0;k<n;++k) {
                double term=(double)a[row*n+k]*b[k*n+col];
                expected+=term; scale+=fabs(term);
            }
            assert(isfinite(c[position]));
            assert(fabs(c[position]-expected) <= 32*FLT_EPSILON*(scale+1));
        }
        for (size_t x=0;x<count;++x) assert(isfinite(c[x]));
        for (size_t x=0;x<16;++x) assert(guarded[x]==12345 && guarded[count+16+x]==12345);
    }
    free(a);free(b);free(ap);free(bp);free(guarded);
}
int main(void) {
    const size_t ns[]={32,64,96,128,160,192,256,288,320,512,544,576,1088,2080,2112};
    for (size_t i=0;i<sizeof(ns)/sizeof(ns[0]);++i)
        for (unsigned seed=1;seed<=2;++seed) check(ns[i],seed);
    puts("SME numerical regression PASS: 15 sizes, 2 seeds, 2 overwrites, guard regions.");
    return 0;
}
