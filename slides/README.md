# SME on Apple M4 Pro

A 22-slide teaching deck in English, based on the repository's measured full
run. Designed for programmers familiar with matrix multiplication; allow
about 25 minutes.

- [PDF](sme-on-apple-m4-pro.pdf): ready to read or present.
- [PowerPoint](sme-on-apple-m4-pro.pptx): editable text, diagrams, and speaker notes.
- [Overview](overview.png): all slides on one image.
- [Speaker notes](speaker_notes.md): detailed explanations and per-slide sources.
- [Chart assets](assets/): PNG and editable SVG charts.
- [Individual previews](previews/): every slide as a PNG.

The PDF and PowerPoint are generated from the same layout. Charts are embedded
images in PowerPoint; their SVG versions and plotting code are provided for
editing. The PDF contains selectable text and vector diagrams.

## Reading order

| Slides | Topic |
| --- | --- |
| 1–5 | SME versus Accelerate, outer products, streaming mode, ZA and scalable vectors |
| 6–9 | Original 32×32 implementation, measurement protocol, baseline results, cost diagnosis |
| 10–16 | Size dispatch, A transpose in ZA, direct B loads, 16×64 kernel, tuple lifetime, blocking and B packing |
| 17–20 | Final throughput, speedup versus Accelerate, improvement over original SME32, selected results |
| 21–22 | Reproduction, scope of the measurements, references and source reading map |

“Original SME32” means this project's original implementation, rather than an
official standard kernel. Accelerate is a vendor library reference; its internal
instruction path is not inspected. All performance charts use
[`examples/m4-pro/`](../examples/m4-pro/). They include packing except where
explicitly marked **compute** or **prepared inputs**.

Speedups use within-round median timing ratios. Shading shows the IQR across
three rounds, not a confidence interval. The deck includes the N=2048 result
where Accelerate remains faster, and treats the N=4096 result as practical parity.
Per-optimization speedups are not claimed: the measurements compare the
complete original and optimized implementations.

## Rebuild without running a benchmark

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r slides/requirements.txt
.venv/bin/python slides/build_slides.py
```

This rebuilds the PPTX, PDF, notes, outline, chart assets and input manifest.
It reads existing results and source files; it never compiles or executes GEMM.
The benchmark itself still requires only the dependencies described in the
root README. Presentation dependencies are separate.

To regenerate the optional overview and slide previews as well:

```sh
.venv/bin/python -m pip install PyMuPDF==2.1
.venv/bin/python slides/build_slides.py --preview
```

Arial is used when available on macOS. PDF generation falls back to Matplotlib's
bundled DejaVu fonts on other platforms; PowerPoint viewers may substitute
Arial according to their installed fonts.

The generator checks throughput against timings, paired speedup medians
against the raw jobs, source hashes against the measured build, the full-suite
shape, and text widths/slide bounds. `data_manifest.json` records input hashes.
The included deck was also rendered and visually inspected.

The explanations describe this M4 Pro implementation and its default full run
(120 jobs, 840 samples). To use another compatible M4 Pro full run:

```sh
.venv/bin/python slides/build_slides.py --results results/run1 --output results/run1/slides
```

Charts, tables, and numerical callouts are derived from the input data. Review
the teaching narrative and speaker notes when replacing the measured run:
some observations describe the bundled snapshot. Changing tuning constants,
hardware, or source code also requires updating the explanations. Outputs in
the chosen slide directory are overwritten; benchmark input files are not.

## Sources

Architecture and API claims cite primary references on the reference slide and
in speaker notes:

- [Arm C Language Extensions](https://arm-software.github.io/acle/main/acle.html)
- [Hello SME microbenchmarks, University Jena](https://scalable.uni-jena.de/opt/sme/micro.html)
- [Apple Accelerate](https://developer.apple.com/accelerate/)
- [Apple single-thread BLAS control](https://developer.apple.com/documentation/accelerate/blas_threading_single_threaded)

External microbenchmark numbers are not used as local benchmark results.
Earlier experimental disassembly for the rejected cross-iteration tuple
pipeline was not retained; that optimization is identified as a tuning
observation recorded in the current source, without an invented ablation chart.
