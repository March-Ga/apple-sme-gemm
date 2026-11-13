#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include "gemm.h"

int accelerate_prepare(void) {
    return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0 &&
           BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED;
}

void gemm_accelerate(size_t n, const float *restrict a, const float *restrict b,
                     float *restrict c) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)n, (int)n, (int)n, 1.0f, a, (int)n, b, (int)n, 0.0f, c, (int)n);
}
