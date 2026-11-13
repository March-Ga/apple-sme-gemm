// Independent full-layout verification, including zero padding at the B tail.
#include "../src/gemm.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check(size_t n) {
    const size_t count = n*n, b_count = sme_packed_b_count(n);
    float *a=malloc(count*sizeof(float)), *b=malloc(count*sizeof(float));
    float *ap=malloc(count*sizeof(float)), *bp=malloc((b_count+1)*sizeof(float));
    assert(a && b && ap && bp);
    for (size_t x=0;x<count;++x) { a[x]=(float)(x+1); b[x]=-(float)(x+1); ap[x]=NAN; }
    for (size_t x=0;x<=b_count;++x) bp[x]=NAN;
    pack_inputs(n,a,b,ap,bp);
    for (size_t row=0;row<n;++row)
        for (size_t col=0;col<n;++col) {
            const size_t pc=col/KC*KC, kc=n-pc<KC ? n-pc : KC;
            const size_t dst=pc*n+(row/16)*16*kc+(col-pc)*16+row%16;
            assert(ap[dst]==a[row*n+col]);
        }
    if (n>DIRECT_B_MAX) {
        const size_t columns=(n+63)/64*64;
        for (size_t row=0;row<n;++row)
            for (size_t col=0;col<columns;++col) {
                const size_t pc=row/KC*KC, kc=n-pc<KC ? n-pc : KC;
                const size_t dst=pc*columns+(col/64)*64*kc+(row-pc)*64+col%64;
                assert(bp[dst]==(col<n ? b[row*n+col] : 0));
            }
    }
    assert(isnan(bp[b_count]));
    free(a); free(b); free(ap); free(bp);
}
int main(void) {
    const size_t cases[]={32,64,96,128,192,288,320,512,544,576,1024,1088,2048,2080,2112};
    for (size_t x=0;x<sizeof(cases)/sizeof(cases[0]);++x) check(cases[x]);
    puts("Packing layout PASS: 15 sizes, every packed value, B padding and end guard.");
    return 0;
}
