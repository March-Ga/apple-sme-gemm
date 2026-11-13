# apple-sme-gemm

A minimal, reproducible comparison of **optimized SME**, **original SME32**
and **Apple Accelerate**. Written in C with intrinsics, using square FP32
matrices and one thread. The kernels were validated on a MacBook Pro M4 Pro;
the project is self-contained.

![M4 Pro throughput and speedup](examples/m4-pro/plots/comparison.png)

## Quick start

Requires macOS 15+, Apple Silicon with SME2 and a 512-bit streaming vector
length, Command Line Tools supporting `apple-m4` (tested with Apple Clang 17),
and Python 3.10+. Building, testing and benchmarking require only system
tools and the Python standard library; plotting additionally requires Matplotlib.

```sh
python3 build.py
python3 check.py
python3 benchmark.py --suite full --output results/run1
python3 plot.py results/run1
```

To install the plotting dependency in a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python plot.py results/run1
```

Output directories must not already exist. For a short run, replace the
benchmark command with:

```sh
python3 benchmark.py --suite quick --rounds 2 --samples 5 --min-sample-ms 20 --output results/quick
```

The Makefile also provides `make build`, `make test`, `make quick`, `make bench`
and `make plot`. Run `make bench plot OUT=results/run2` to build, benchmark
and plot. Select an interpreter with `PYTHON=.venv/bin/python`.

## Implementations

| Backend | Strategy |
| --- | --- |
| `sme32` | Original 32×32 kernel, four ZA tiles, A16/B16 packing, K blocking of 256 |
| `sme_opt` | Up to N=512: 32×32 kernel, A transposed in ZA, original row-major B without packing. Larger sizes: 16×64 kernel, SME2 four-vector loads, A16/B64 packing and blocking K=2048, M=256, N=512 |
| `accelerate` | `cblas_sgemm`, with BLAS explicitly set to one thread |

For N>512 that is not divisible by 64, the optimized backend uses the original
SME32 path. The SME kernels neither pad the matrices nor delegate to Accelerate.
Supported dimensions are multiples of 32 from 32 to 8192. The JSON fields
`kernel_variant` and `packing_layout` identify the selected path.
Packing large B matrices uses blocks of 256 columns and 32 rows. The kernel
loop is unrolled four times, without the earlier software pipeline that
generated extra register copies with Clang 17.

## What the plots measure

| Mode | Timed work |
| --- | --- |
| `total` | Backend-specific packing plus the complete product; Accelerate receives row-major inputs |
| `compute` | Product only, with inputs already prepared; the small optimized kernel keeps B row-major |

Accelerate appears only in `total`, because it does not expose the packing
interface used by our kernels. The main comparison **includes packing**.
Allocation, initialization, process startup, calibration and numerical
verification are outside the timer. Buffers are aligned to 64 bytes.

- **Throughput**: `2*N³ / time_in_seconds / 10⁹`, in GFLOP/s.
- **Speedup**: `baseline_time / backend_time`. Values above 1 mean the backend
  is faster. The baseline is Accelerate in `total` and SME32 in `compute`.
- Speedups pair backend medians within each round, then report the median
  of those ratios. Shading shows the interquartile range (IQR) across rounds,
  not a confidence interval. The ratio of pooled medians can differ.

The full suite uses N=32…4096, three rounds and seven samples per job. Cases
and backends are randomized with a reproducible seed. Each sample averages
a batch calibrated to approximately 30 ms or longer; it does not measure
individual-call tail latency. Inputs are reused, measuring steady-state work.
`USER_INITIATED` QoS favors fast cores, but macOS controls scheduling. Keep
power source, energy mode and competing workloads consistent between runs.

The comparison includes packing and algorithm differences: it measures
implementations, not the isolated effect of the ISA. It does not measure
the chip's maximum multicore throughput.

## Results and validation

`plots/comparison.png`, `.svg` and `.pdf` show throughput and speedup for
both modes. Each run directory also contains:

| File | Contents |
| --- | --- |
| `summary.csv` | Median times, quartiles and throughput per backend |
| `speedups.csv` | Paired speedups against SME32 and Accelerate |
| `raw.jsonl`, `samples.csv` | All raw job results and timing samples |
| `verification.csv` | Numerical checks and input fingerprints |
| `environment.json` | Hardware, compiler, options and source/binary hashes |
| `status.json` | Suite completion status |

`Ctrl+C` preserves completed jobs and partial reports. If a job fails,
`failed_job.json` records the command needed to reproduce it.

`check.py` builds and runs tests with AddressSanitizer and UndefinedBehaviorSanitizer:
complete packing checks at 12 dimensions, and GEMM checks at 15 dimensions
with two seeds, two output overwrites, output guards and a scalar double
reference. It checks every element through N=192 and 512 entries at larger
sizes. Python tests cover pairing, metrics, reports without timings and dry runs.
Each benchmark job also checks output before and after sampling: every element
through N=128, otherwise 256 entries, plus a finiteness check on all output.

The build inspects disassembly for the four ZA tiles and the optimized
kernel's multi-vector loads. The runner rejects sources or binaries that
differ from the build manifest: run `python3 build.py` after changing C code.

For custom sizes or numerical verification only:

```sh
python3 benchmark.py --sizes 32,64,96,320,544,1088,2080,2112 --verify-only --output results/checks
python3 benchmark.py --sizes 512,1024 --modes total --output results/custom
python3 benchmark.py --suite full --dry-run
```

## Validated example

[`examples/m4-pro/`](examples/m4-pro/) contains a full run performed from
this repository: raw data, metadata and plots. These results are not universal;
small differences should be interpreted alongside their variability.
The optimized implementation does not beat Accelerate at every size.

Local measurements on an M4 Pro with 48 GB RAM and Apple Clang 17, including packing:

| N | SME32 GFLOP/s | Optimized GFLOP/s | Accelerate GFLOP/s | Optimized speedup over Accelerate |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 640 | 1539 | 1407 | 1.09× |
| 512 | 1342 | 1797 | 1792 | 1.00× |
| 1024 | 1296 | 1702 | 1668 | 1.02× |
| 2048 | 1095 | 1602 | 1696 | 0.95× |
| 4096 | 945 | 1576 | 1571 | 1.01× |

All 120 jobs and 840 samples passed verification. At N=2048, the optimized
kernel is approximately 6% slower than Accelerate in this run; at N=4096,
performance is effectively equal. Differences of a few percentage points
do not establish a meaningful advantage without considering variability.

Kernels live in `src/sme_opt.c` and `src/sme32.c`; packing is implemented in
`src/pack*.c` and `src/sme_pack.c`. `main.c` handles timing and verification.
The four Python scripts handle building, testing, benchmarking and plotting.
No Eigen or other external BLAS dependency is required.
