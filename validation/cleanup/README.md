# Cleanup validation

This verifies the repository simplification against the optimized implementation
at commit `12a449ed8df591d8946f6581efbc32581d2c3d28`. Historical implementation
source and executables are not included in this worktree.

## Protocol

The two revisions were built with the same Apple Clang 17 flags on the same
M4 Pro. Every timed call includes packing and the complete product. A temporary
external harness randomized revision/backend order within each size and round,
using identical inputs, one thread, three warmups, three rounds, seven samples,
and a batch target of at least approximately 30 ms. Accelerate was measured for
both revisions as a control. All input fingerprints matched and every job
passed numerical verification.

There were **120 jobs and 840 samples** across eight standard sizes and two
32-column-tail sizes. `raw.jsonl` records every job and all batch-average
measurements; `environment.json` records both measured builds and their hashes.
Only the optimized SME path and Accelerate were measured in the reference.

## Results

Ratios are medians of within-round timing ratios. **Above 1 means the cleaned
code is faster.** The control column is reference Accelerate time divided by
clean-build Accelerate time. The normalized column divides the SME ratio by
the corresponding control ratio within each round, then takes the median.
This is a diagnostic for shared timing variation, not a confidence interval
or proof of statistical equivalence.

| N | SME cleaned / reference | Accelerate control | Normalized SME ratio |
| ---: | ---: | ---: | ---: |
| 32 | 0.996× | 1.011× | 0.985× |
| 64 | 1.002× | 1.000× | 0.999× |
| 128 | 0.999× | 1.004× | 0.995× |
| 256 | 1.000× | 0.999× | 1.002× |
| 512 | 1.001× | 1.000× | 0.999× |
| 544 | 1.081× | 0.999× | 1.082× |
| 1024 | 0.999× | 1.001× | 0.997× |
| 2048 | 1.009× | 0.990× | 1.001× |
| 2080 | 1.342× | 1.018× | 1.318× |
| 4096 | 1.002× | 0.995× | 1.006× |

Across the eight standard sizes, the paired direct SME ratios range from
**0.996× to 1.009×**: no material slowdown is observed in this experiment.
The final tail handling improves N=544 by about **1.08×** and N=2080 by about
**1.34×**, relative to the paths selected by the previous implementation.
The normalized standard-size ratios span approximately 0.985×–1.006×.
Individual differences of a few percent should be interpreted with variability.

`comparison.csv` also includes pooled median latencies. Their ratios can differ
from paired medians. The current standalone full run is in
[`examples/m4-pro/`](../../examples/m4-pro/); it contains only the two current
backends, 48 jobs and 336 samples, and supplies every current plot and slide.

## Correctness and structural checks

- ASan/UBSan: every packed entry and padded B entry at 15 sizes, with an end guard.
- ASan/UBSan: both SME and Accelerate at 15 sizes, two seeds and two output
  overwrites; scalar double reference, explicit tail/edge positions and output guards.
- Five reporting/CLI regression tests, including missing paired rounds and verification-only output.
- Additional verification-only runs for both backends at N=32,544,2080,4096,8160,8192:
  **12 completed jobs**, covering the supported upper bound and its 32-column tail.
- Disassembly audit: all four ZA destinations and SME2 grouped loads present.
  The final executable exposes only `gemm_sme` and `gemm_accelerate` as GEMM entry points.

`tests.txt` contains the sanitizer/reporting test output.
`edge_verification.jsonl` and `edge_status.json` contain the additional checks.
`status.json` confirms completion of the interleaved timing experiment.
These are validation records, not additional runnable implementation variants.

The historical reference can be recovered from Git for a future comparison;
it is deliberately absent from the simplified worktree. The before/after
harness was temporary and is not needed to build, test, benchmark or plot the
final two backends. The raw records preserve the commands' essential settings
and all timing samples for independent recalculation.
