// Build (no Eigen): python3 build.py
// Run one case: ./build/sme_bench --backend sme32 --mode total --size 256
#include "src/gemm.h"
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

typedef void (*Gemm)(size_t, const float *, const float *, float *);
typedef void (*Pack)(size_t, const float *, const float *, float *, float *);
typedef struct {
    size_t n;
    unsigned samples, warmup;
    double target_ms;
    uint64_t seed;
    const char *backend, *mode;
    int verify_only;
} Options;
typedef struct {
    size_t n;
    float *a, *b, *c, *ap, *bp;
    Gemm compute;
    Pack pack;
    int total, apple, direct_b;
} Job;
typedef struct { size_t entries; double max_error; } Verification;

static void fail(const char *message) {
    fprintf(stderr, "sme_bench: %s\n", message);
    exit(1);
}

static uint64_t integer(const char *text) {
    if (!*text || *text == '-') fail("Expected an unsigned integer");
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end) fail("Invalid integer argument");
    return (uint64_t)value;
}

static Options parse(int argc, char **argv) {
    Options o = {.n=256, .samples=7, .warmup=3, .target_ms=30,
                 .seed=20261113, .backend="sme_opt", .mode="total", .verify_only=0};
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--help")) {
            puts("Square FP32 GEMM, C=A*B, one thread; no Eigen.\n"
                 "--backend sme32|sme_opt|accelerate\n"
                 "--mode compute|total\n"
                 "--size N (32..8192, multiple of 32)\n"
                 "--samples N --warmup N --min-sample-ms N --seed N\n"
                 "--verify-only (compute and check, without timing samples)\n"
                 "Prefer python3 benchmark.py to compare paired rounds.");
            exit(0);
        }
        if (!strcmp(key, "--verify-only")) { o.verify_only = 1; continue; }
        if (++i == argc) fail("Missing option value");
        const char *value = argv[i];
        if (!strcmp(key, "--backend")) o.backend = value;
        else if (!strcmp(key, "--mode")) o.mode = value;
        else if (!strcmp(key, "--seed")) o.seed = integer(value);
        else if (!strcmp(key, "--size")) {
            const uint64_t number = integer(value);
            if (number < 32 || number > 8192 || number%32) fail("N must be a multiple of 32 in [32,8192]");
            o.n = (size_t)number;
        } else if (!strcmp(key, "--samples") || !strcmp(key, "--warmup")) {
            const uint64_t number = integer(value);
            if (number < 1 || number > 100) fail("samples/warmup must be in [1,100]");
            if (!strcmp(key, "--samples")) o.samples = (unsigned)number;
            else o.warmup = (unsigned)number;
        } else if (!strcmp(key, "--min-sample-ms")) {
            char *end = NULL;
            o.target_ms = strtod(value, &end);
            if (!end || *end || !isfinite(o.target_ms) || o.target_ms < 1 || o.target_ms > 10000)
                fail("min-sample-ms must be in [1,10000]");
        } else fail("Unknown option");
    }
    if (strcmp(o.mode, "compute") && strcmp(o.mode, "total")) fail("Unknown mode");
    if (strcmp(o.backend, "sme32") && strcmp(o.backend, "sme_opt") && strcmp(o.backend, "accelerate")) fail("Unknown backend");
    if (!strcmp(o.backend, "accelerate") && !strcmp(o.mode, "compute"))
        fail("Accelerate is a total-mode reference; it has no public prepacked interface");
    return o;
}

static int has_feature(const char *name) {
    int supported = 0;
    size_t bytes = sizeof(supported);
    return sysctlbyname(name, &supported, &bytes, NULL, 0) == 0 && supported;
}

static float *allocate(size_t entries) {
    void *memory = NULL;
    if (posix_memalign(&memory, 64, entries*sizeof(float))) fail("Allocation failed");
    return memory;
}

static uint64_t random_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z = (z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}

static uint64_t initialize(Job *job, uint64_t seed) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const size_t count = job->n*job->n;
    float *inputs[] = {job->a, job->b};
    for (unsigned input = 0; input < 2; ++input)
        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = (uint32_t)(random_next(&seed)>>40);
            // Subtract in a signed type: size_t arithmetic would underflow.
            inputs[input][i] = (float)((int32_t)bits-8388608)*(1.0f/8388608.0f);
            hash = (hash^bits)*UINT64_C(1099511628211);
        }
    for (size_t i = 0; i < count; ++i) job->c[i] = NAN;
    return hash;
}

static void execute(Job *job) {
    if (job->apple) job->compute(job->n, job->a, job->b, job->c);
    else {
        if (job->total) job->pack(job->n, job->a, job->b, job->ap, job->bp);
        job->compute(job->n, job->ap, job->direct_b ? job->b : job->bp, job->c);
    }
}

// Independent scalar double-precision reference, never used as the speed baseline.
static Verification verify(const Job *job, uint64_t seed) {
    const size_t n = job->n, count = n*n;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(job->c[i])) fail("Output contains NaN or infinity");
    const int full = n <= 128;
    size_t positions[256];
    size_t checks = full ? count : 0;
    if (!full) {
        // Cover corners and all four 16x16 quadrants before random positions.
        const size_t boundary[] = {0, 15, 16, 31, 32, n/2, n-1};
        for (size_t r = 0; r < sizeof(boundary)/sizeof(boundary[0]); ++r)
            for (size_t c = 0; c < sizeof(boundary)/sizeof(boundary[0]); ++c)
                positions[checks++] = boundary[r]*n+boundary[c];
        while (checks < 256) {
            const size_t position = random_next(&seed)%count;
            int duplicate = 0;
            for (size_t i = 0; i < checks; ++i) duplicate |= positions[i] == position;
            if (!duplicate) positions[checks++] = position;
        }
    }
    double maximum = 0;
    const double neps = (double)n*FLT_EPSILON;
    const double gamma = neps/(1-neps);
    for (size_t check = 0; check < checks; ++check) {
        const size_t position = full ? check : positions[check];
        const size_t row = position/n, column = position%n;
        double expected = 0, scale = 0;
        for (size_t k = 0; k < n; ++k) {
            const double term = (double)job->a[row*n+k]*(double)job->b[k*n+column];
            expected += term;
            scale += fabs(term);
        }
        const double error = fabs((double)job->c[position]-expected);
        if (error > 4*gamma*scale+8*FLT_EPSILON) {
            fprintf(stderr, "Mismatch at C[%zu,%zu]: got %.9g, reference %.17g\n",
                    row, column, job->c[position], expected);
            fail("Numerical verification failed");
        }
        maximum = fmax(maximum, error/fmax(scale, 1e-30));
    }
    return (Verification){.entries=checks, .max_error=maximum};
}

static uint64_t nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now)) fail("Cannot read monotonic clock");
    return (uint64_t)now.tv_sec*UINT64_C(1000000000)+(uint64_t)now.tv_nsec;
}

static double timed(Job *job, uint64_t repetitions) {
    const uint64_t start = nanoseconds();
    for (uint64_t r = 0; r < repetitions; ++r) execute(job);
    return (double)(nanoseconds()-start)*1e-6;
}

int main(int argc, char **argv) {
    const Options o = parse(argc, argv);
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0))
        fail("Cannot set requested thread QoS");
    const int is_opt = !strcmp(o.backend, "sme_opt");
    const int is_sme = !strcmp(o.backend, "sme32") || is_opt;
    size_t vector_lanes = 0;
    if (is_sme) {
        if (!has_feature("hw.optional.arm.FEAT_SME")) fail("CPU/OS does not expose SME");
        if (is_opt && !has_feature("hw.optional.arm.FEAT_SME2")) fail("Optimized backend requires SME2");
        vector_lanes = sme_vector_lanes();
        if (vector_lanes != PANEL) fail("This kernel requires a 512-bit streaming vector length");
    }
    Job job = {.n=o.n, .total=!strcmp(o.mode, "total"), .apple=!strcmp(o.backend, "accelerate")};
    job.direct_b = is_opt && o.n <= DIRECT_B_MAX;
    job.pack = is_opt ? pack_inputs_opt : pack_inputs;
    job.compute = !strcmp(o.backend, "sme32") ? gemm_sme32 :
                  is_opt ? gemm_sme_opt : gemm_accelerate;
    if (job.apple && !accelerate_prepare()) fail("Cannot force Accelerate BLAS to one thread");
    const size_t count = o.n*o.n;
    job.a = allocate(count); job.b = allocate(count); job.c = allocate(count);
    if (!job.apple) { job.ap = allocate(count); job.bp = allocate(count); }
    const uint64_t fingerprint = initialize(&job, o.seed);
    if (!job.apple && !job.total) job.pack(o.n, job.a, job.b, job.ap, job.bp);

    const unsigned warmups = o.verify_only ? 1 : o.warmup;
    for (unsigned w = 0; w < warmups; ++w) execute(&job);
    verify(&job, o.seed);
    uint64_t repetitions = o.verify_only ? 0 : 1;
    double samples[100];
    if (!o.verify_only) {
        for (unsigned attempt = 0; attempt < 12; ++attempt) {
            const double elapsed = timed(&job, repetitions);
            if (elapsed >= o.target_ms || repetitions >= 1048576) break;
            const double factor = fmin(8, fmax(2, o.target_ms/fmax(elapsed, 1e-6)*1.1));
            repetitions = (uint64_t)fmin(1048576, ceil((double)repetitions*factor));
        }
        volatile float consumed = 0;
        for (unsigned s = 0; s < o.samples; ++s) {
            samples[s] = timed(&job, repetitions)/(double)repetitions;
            consumed = job.c[((size_t)s*997)%count];
        }
        (void)consumed;
    }
    const Verification checked = verify(&job, o.seed);
    printf("{\"backend\":\"%s\",\"mode\":\"%s\",\"size\":%zu,\"dtype\":\"fp32\","
           "\"threads\":1,\"kc\":%d,\"panel\":%d,\"streaming_fp32_lanes\":%zu,"
           "\"seed\":%" PRIu64 ",\"input_fingerprint\":\"%016" PRIx64 "\","
           "\"verified_entries\":%zu,\"max_scaled_error\":%.17g,\"verification_only\":%s,"
           "\"warmup_calls\":%u,\"batch_calls\":%" PRIu64 ",\"min_sample_ms\":%.17g,\"samples_ms\":[",
           o.backend, o.mode, o.n, is_opt && (job.direct_b || o.n%64 == 0) ? OPT_KC : KC, PANEL, vector_lanes, o.seed, fingerprint,
           checked.entries, checked.max_error, o.verify_only ? "true" : "false", warmups,
           repetitions, o.target_ms);
    if (!o.verify_only)
        for (unsigned s = 0; s < o.samples; ++s) printf("%s%.17g", s ? "," : "", samples[s]);
    printf("],\"kernel_variant\":\"%s\",\"packing_layout\":\"%s\"}\n",
           is_opt ? (job.direct_b ? "32x32-direct-b" : o.n%64 ? "32x32-fallback" : "16x64-blocked") : o.backend,
           job.apple ? "native-row-major" : job.direct_b ? "A16-original-B" :
           is_opt && o.n%64 == 0 ? "A16-B64" : "A16-B16");
    free(job.a); free(job.b); free(job.c); free(job.ap); free(job.bp);
    return 0;
}
