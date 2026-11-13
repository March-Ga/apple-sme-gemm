#include "gemm.h"
#include <arm_sme.h>

size_t sme_vector_lanes(void) { return svcntsw(); }
