#!/usr/bin/env python3
"""Build the teaching deck from saved benchmark data; never run a benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/apple-sme-slide-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/apple-sme-slide-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.util import Pt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

W, H = 960, 540
BG, INK, MUTED = "#F7F8FB", "#172338", "#59667C"
TEAL, PINK, ORANGE, BLUE = "#007F86", "#DB2777", "#C76B13", "#3255C5"
GRID, WHITE, DARK = "#DBE2EC", "#FFFFFF", "#14243A"
COLORS = {"sme32": TEAL, "sme_opt": PINK, "accelerate": ORANGE}
LABELS = {"sme32": "Original SME32", "sme_opt": "Optimized SME", "accelerate": "Accelerate"}
URLS = {
    "ACLE": "https://arm-software.github.io/acle/main/acle.html",
    "Jena": "https://scalable.uni-jena.de/opt/sme/micro.html",
    "Apple": "https://developer.apple.com/accelerate/",
    "Threads": "https://developer.apple.com/documentation/accelerate/blas_threading_single_threaded",
}


def register_fonts():
    """Use local Arial when available, with a portable bundled-font fallback."""
    supplemental = Path("/System/Library/Fonts/Supplemental")
    fallback = Path(matplotlib.get_data_path()) / "fonts/ttf"
    choices = {
        "Body": (supplemental / "Arial.ttf", fallback / "DejaVuSans.ttf"),
        "Bold": (supplemental / "Arial Bold.ttf", fallback / "DejaVuSans-Bold.ttf"),
        "Mono": (supplemental / "Courier New.ttf", fallback / "DejaVuSansMono.ttf"),
    }
    for name, paths in choices.items():
        pdfmetrics.registerFont(TTFont(name, str(next(p for p in paths if p.exists()))))


def shade(hex_color, fraction=.10):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c*fraction+255*(1-fraction)):02x}" for c in rgb)


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lo = int(position)
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


class Data:
    def __init__(self, repo, result):
        self.repo, self.result = repo, result
        self.summary = load_csv(result / "summary.csv")
        self.speedups = load_csv(result / "speedups.csv")
        self.env = json.loads((result / "environment.json").read_text())
        self.status = json.loads((result / "status.json").read_text())
        self.sizes = sorted({int(r["size"]) for r in self.summary})
        expected = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.sizes != expected or not self.status["complete"]:
            raise ValueError("This teaching deck requires a complete full suite, N=32..4096.")
        self.index = {(int(r["size"]), r["mode"], r["backend"]): r for r in self.summary}
        self.ratios = {(int(r["size"]), r["mode"], r["baseline"], r["backend"]): r
                       for r in self.speedups}
        for n in self.sizes:
            for mode, backends in (("total", COLORS), ("compute", ("sme32", "sme_opt"))):
                for backend in backends:
                    r = self.index[n, mode, backend]
                    calculated = 2*n**3 / float(r["median_ms"]) / 1e6
                    if not math.isclose(calculated, float(r["gflops"]), rel_tol=1e-10):
                        raise ValueError("Throughput does not match the timing data")
        # Validate paired ratios against raw jobs, rather than trusting a chart CSV.
        records = [json.loads(line) for line in (result / "raw.jsonl").read_text().splitlines()]
        paired = {(r["size"], r["mode"], r["round"], r["backend"]):
                  statistics.median(r["samples_ms"]) for r in records}
        for (n, mode, baseline, backend), row in self.ratios.items():
            ratios = [paired[n, mode, r, baseline]/paired[n, mode, r, backend]
                      for r in range(1, int(row["paired_rounds"])+1)]
            for field, expected in (("speedup_median", statistics.median(ratios)),
                                    ("speedup_p25", quantile(ratios, .25)),
                                    ("speedup_p75", quantile(ratios, .75))):
                if not math.isclose(expected, float(row[field]), rel_tol=1e-10):
                    raise ValueError(f"Paired speedup/IQR does not match raw data: {field}")
        self.jobs = len(records)
        self.samples = sum(len(r["samples_ms"]) for r in records)
        if self.jobs != 120 or self.samples != 840 or self.env["cpu"] != "Apple M4 Pro":
            raise ValueError("This deck describes the default 120-job / 840-sample M4 Pro full suite.")
        # The explanations below are specific to the implementation being taught.
        for file, pattern in (("src/gemm.h", r"OPT_KC\s*=\s*2048"),
                              ("src/gemm.h", r"OPT_MC\s*=\s*256"),
                              ("src/gemm.h", r"OPT_NC\s*=\s*512"),
                              ("src/gemm.h", r"DIRECT_B_MAX\s*=\s*512")):
            if not re.search(pattern, (repo/file).read_text()):
                raise ValueError("Update slide explanations after changing tuning constants")
        for name, digest in self.env["build"]["source_sha256"].items():
            if sha(repo/name) != digest:
                raise ValueError(f"Saved run does not match source: {name}")

    def value(self, n, backend, mode="total", field="gflops"):
        return float(self.index[n, mode, backend][field])

    def ratio(self, n, baseline="accelerate", mode="total", field="speedup_median", backend="sme_opt"):
        return float(self.ratios[n, mode, baseline, backend][field])


class Slide:
    def __init__(self, section, title, subtitle="", source="", notes=""):
        self.items, self.notes, self.title = [], notes, title
        self.rect(0, 0, W, H, BG)
        self.text(48, 26, 864, section.upper(), 10, TEAL, bold=True)
        self.text(48, 53, 864, title, 29, bold=True)
        if subtitle:
            self.text(48, 97, 864, subtitle, 13, MUTED)
        self.line(48, 493, 912, 493, GRID)
        self.text(48, 504, 825, source, 8.5, MUTED)

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, radius=0):
        self.items.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill,
                               stroke=stroke, radius=radius))

    def line(self, x, y, x2, y2, color=GRID, width=1):
        self.items.append(dict(kind="line", x=x, y=y, x2=x2, y2=y2, color=color, width=width))

    def text(self, x, y, w, text, size=18, color=INK, bold=False, mono=False, link=None):
        font = "Mono" if mono else "Bold" if bold else "Body"
        # Wrap once using measured glyph widths; use exactly these lines in both exports.
        lines = []
        for para in str(text).split("\n"):
            if mono:
                lines.append(para)
                continue
            words = para.split(" ")
            current = ""
            for word in words:
                proposed = current + (" " if current else "") + word
                if current and pdfmetrics.stringWidth(proposed, font, size) > w-2:
                    lines.append(current)
                    current = word
                else:
                    current = proposed
            lines.append(current)
        h = size*1.24*len(lines)+5
        self.items.append(dict(kind="text", x=x, y=y, w=w, h=h, lines=lines,
                               size=size, color=color, font=font, link=link))
        return h

    def arrow(self, x, y, w, color=TEAL):
        self.items.append(dict(kind="arrow", x=x, y=y, w=w, h=13, color=color))

    def image(self, x, y, w, h, path):
        self.items.append(dict(kind="image", x=x, y=y, w=w, h=h, path=str(path)))

    def card(self, x, y, w, h, heading, body, color=TEAL, size=17):
        self.rect(x, y, w, h, WHITE, radius=8)
        self.rect(x, y, 5, h, color)
        self.text(x+20, y+18, w-38, heading, 19, color, bold=True)
        self.text(x+20, y+55, w-38, body, size)

    def band(self, text, color=TEAL):
        self.rect(48, 445, 864, 35, shade(color), radius=5)
        self.text(62, 454, 836, text, 14, color, bold=True)

    def code(self, x, y, w, text, size=12, h=None):
        content = textwrap.dedent(text).strip("\n")
        count = len(content.splitlines())
        self.rect(x, y, w, h or count*size*1.24+30, DARK, radius=6)
        self.text(x+16, y+14, w-32, content, size, "#E9F1FC", mono=True)


def matrix(s, x, y, values, colors=None, cell=40, size=17):
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            color = colors[i][j] if colors else shade(TEAL)
            s.rect(x+j*cell, y+i*cell, cell-2, cell-2, color)
            s.text(x+j*cell+9, y+i*cell+8, cell-15, str(value), size)


def charts(data, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.facecolor": BG,
                         "figure.facecolor": BG, "axes.facecolor": BG})
    paths = {}
    sizes = data.sizes

    def finish(fig, name):
        fig.tight_layout(pad=1.1)
        for ext in ("png", "svg"):
            target = output/f"{name}.{ext}"
            fig.savefig(target, dpi=180)
            if ext == "svg":
                target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
        paths[name] = output/f"{name}.png"
        plt.close(fig)

    def xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Square matrix dimension N")
        ax.grid(axis="y", color=GRID, alpha=.7)
        ax.set_axisbelow(True)

    def series(ax, backend, mode="total"):
        ax.plot(sizes, [data.value(n, backend, mode)/1000 for n in sizes],
                "o-", lw=2.4, ms=5, color=COLORS[backend], label=LABELS[backend])

    def ratio_series(ax, backend, baseline="accelerate", mode="total", label=None, color=None):
        color = color or COLORS[backend]
        vals = [data.ratio(n, baseline, mode, backend=backend) for n in sizes]
        lo = [data.ratio(n, baseline, mode, "speedup_p25", backend) for n in sizes]
        hi = [data.ratio(n, baseline, mode, "speedup_p75", backend) for n in sizes]
        ax.plot(sizes, vals, "o-", lw=2.4, ms=5, color=color, label=label or LABELS[backend])
        ax.fill_between(sizes, lo, hi, color=color, alpha=.12)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})
    for backend in ("sme32", "accelerate"):
        series(axes[0], backend)
    axes[0].set_ylabel("Throughput (TFLOP/s)")
    axes[0].set_ylim(0, 2.02)
    axes[0].legend(fontsize=11, loc="lower right")
    ratio_series(axes[1], "sme32")
    axes[1].axhline(1, color=ORANGE, ls="--", label="Accelerate = 1×")
    axes[1].set_ylabel("Original / Accelerate speedup (×)")
    axes[1].set_ylim(0, 1.12)
    axes[1].legend(fontsize=10, loc="lower right")
    for ax in axes:
        xaxis(ax)
    finish(fig, "baseline")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for ax, n, factor, unit in zip(axes, (128, 4096), (1000, 1), ("µs", "ms")):
        for i, backend in enumerate(("sme32", "sme_opt")):
            vals = [data.value(n, backend, m, "median_ms")*factor for m in ("compute", "total")]
            bars = ax.bar([i*3, i*3+1], vals, color=[shade(COLORS[backend], .45), COLORS[backend]], width=.8)
            ax.bar_label(bars, fmt="%.2f", fontsize=11, padding=4)
        ax.set_xticks([0, 1, 3, 4], ["Compute", "Total", "Compute", "Total"])
        ax.set_ylabel(f"Median time ({unit}); lower is better")
        ax.set_title(f"N = {n}", fontsize=14, fontweight="bold")
        ax.text(.20, -.23, "Original SME32", color=TEAL, transform=ax.transAxes, ha="center", fontsize=11)
        ax.text(.78, -.23, "Optimized SME", color=PINK, transform=ax.transAxes, ha="center", fontsize=11)
        ax.set_ylim(0, max(data.value(n, b, "total", "median_ms")*factor for b in ("sme32", "sme_opt"))*1.22)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    finish(fig, "diagnosis")

    fig, ax = plt.subplots(figsize=(8.4, 4.25))
    for backend in COLORS:
        series(ax, backend)
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_ylim(0, 2.08)
    ax.legend(fontsize=11, loc="lower right")
    xaxis(ax)
    finish(fig, "throughput")

    fig, ax = plt.subplots(figsize=(8.4, 4.25))
    for backend in ("sme32", "sme_opt"):
        ratio_series(ax, backend)
    ax.axhline(1, color=ORANGE, ls="--", lw=1.5, label="Accelerate = 1×")
    ax.set_ylabel("Speedup vs Accelerate (×)")
    ax.set_ylim(0, 3.25)
    ax.legend(fontsize=11, loc="upper right")
    xaxis(ax)
    finish(fig, "speedup")

    fig, ax = plt.subplots(figsize=(11.5, 4.1))
    ratio_series(ax, "sme_opt", "sme32", "total", "Packing + computation", PINK)
    ratio_series(ax, "sme_opt", "sme32", "compute", "Prepared inputs", BLUE)
    ax.axhline(1, color=MUTED, ls="--", lw=1)
    ax.set_ylabel("Optimized / original speedup (×)")
    ax.set_ylim(.85, 4.65)
    ax.legend(fontsize=12, loc="upper right")
    xaxis(ax)
    finish(fig, "improvement")
    return paths


def make_slides(d, assets):
    deck = []
    data_source = "Measured: examples/m4-pro/ · square FP32 · one calling thread · packing included unless stated"

    def new(section, title, subtitle="", source=data_source, notes=""):
        s = Slide(section, title, subtitle, source, notes)
        deck.append(s)
        return s

    s = new("Apple SME GEMM / a measured case study", "From outer products to competitive GEMM",
            "Understanding SME, diagnosing a 32×32 baseline, and tuning for Apple M4 Pro.",
            notes="Audience: programmers familiar with loops and matrix multiplication; no SME knowledge assumed. Allow about 25 minutes. The optimized implementation was measured against the original SME32 implementation and Apple's Accelerate library. All charts use the bundled full run, not estimated peak numbers. The central lesson is to optimize operand delivery, packing, compiler output and blocking together.")
    s.text(50, 153, 560, "A matrix instruction is powerful.\nA fast matrix product also needs\nthe right data movement.", 31, bold=True)
    s.card(662, 147, 248, 126, f"{d.ratio(4096, 'sme32'):.2f}×", "over original SME32\nat N=4096, total", PINK, 16)
    s.card(662, 289, 248, 126, "≈ Accelerate", f"at N=4096\n{d.value(4096, 'sme_opt')/1000:.2f} TFLOP/s, total", ORANGE, 16)
    s.text(50, 335, 548, "01  Understand the hardware interface\n02  Find the cost beyond the multiply\n03  Tune, verify, and compare honestly", 18, MUTED)
    s.band("A real single-thread FP32 benchmark, with reproducible data and source.")

    s = new("01 / The concepts", "SME is an instruction set; Accelerate is a library",
            "The comparison is between implementations using a common GEMM problem.",
            "Sources: Arm ACLE [1] · Apple Accelerate [3] · src/accelerate.c",
            "SME means Scalable Matrix Extension, not scalable extendable matrices. It is an Arm architectural extension exposed through machine instructions and C intrinsics. Accelerate is Apple's framework, including the BLAS interface used here. Apple documents runtime selection of processor-appropriate instructions; this experiment does not inspect the internal cblas_sgemm instruction path. Therefore Accelerate is a vendor library reference, not an SME-disabled control. [1] " + URLS["ACLE"] + " [3] " + URLS["Apple"])
    s.card(48, 147, 414, 229, "SME: Scalable Matrix Extension", "Arm instruction-set extension\nStreaming execution mode\nZ vectors and predicates\nZA matrix accumulators", TEAL, 19)
    s.card(482, 147, 430, 229, "Apple Accelerate", "A framework with BLAS and LAPACK\ncblas_sgemm computes C = A × B\nLibrary chooses its internal path\nOne BLAS thread in this benchmark", ORANGE, 19)
    s.text(61, 396, 835, "SME2 is an extension of SME; the optimized code uses SME2 multi-vector loads.", 17, MUTED)
    s.band("We measure complete implementations, including their different packing strategies.")

    s = new("01 / The operation", "One outer product updates many output values",
            "Matrix multiplication is a sum of outer products: C += A[:, k] × B[k, :].",
            "Sources: Arm ACLE [1] · Hello SME microbenchmarks [2] · numerical example constructed for this deck",
            "Walk through one k value. Select a column of A and a row of B. Multiply each selected A value by each selected B value: a two-element column times a three-element row creates a 2×3 update. Add that update to C. Repeat for every k to obtain the full matrix product. FMOPA carries out a predicated outer-product accumulate into ZA; it does not complete an arbitrary N×N GEMM in one instruction. Example arithmetic is exact integer arithmetic shown as a teaching aid. [1] " + URLS["ACLE"] + " [2] " + URLS["Jena"])
    s.text(70, 147, 170, "A[:, k]", 19, TEAL, bold=True)
    matrix(s, 95, 202, [[2], [3]], cell=58, size=23)
    s.text(179, 235, 45, "×", 30, MUTED)
    s.text(248, 147, 240, "B[k, :]", 19, BLUE, bold=True)
    matrix(s, 245, 230, [[4, 5, 6]], [[shade(BLUE)]*3], cell=58, size=23)
    s.arrow(450, 253, 56)
    s.text(548, 147, 320, "Update added to C", 19, PINK, bold=True)
    matrix(s, 560, 202, [[8, 10, 12], [12, 15, 18]], [[shade(PINK)]*3]*2, cell=65, size=23)
    s.text(60, 369, 820, "On this M4 Pro: 16 FP32 lanes × 16 lanes = 256 output updates per FMOPA.", 19)
    s.band("256 fused multiply-adds = 512 nominal FLOPs per FP32 FMOPA.")

    s = new("01 / Registers and scale", "ZA holds four logical FP32 tiles on this M4 Pro",
            "Scalable means the register dimensions depend on the streaming vector length (SVL).",
            "Sources: Arm ACLE [1] · Hello SME [2] · src/runtime.c · examples/m4-pro/environment.json",
            "This implementation checks the streaming vector length at runtime and requires 16 FP32 lanes, equivalent to 512 bits. With a 64-byte SVL, total ZA storage is 64×64 bytes = 4096 bytes. In the FP32 view this is four 16×16 tiles, each holding 256 floats or 1024 bytes. The colored boxes are logical tile views; do not interpret their placement as physical SRAM geometry. Predicates select active lanes. Our dimensions are multiples of 32, so full-lane predicates are used. [1] " + URLS["ACLE"] + " [2] " + URLS["Jena"])
    s.text(53, 144, 420, "A Z vector: 512 bits = 16 FP32 values", 19, bold=True)
    for i in range(16):
        s.rect(54+i*25, 186, 23, 35, shade(TEAL, .12 if i%2 else .23))
    s.text(54, 243, 406, "SVL / 32 = 16 lanes\nZA storage = 64 × 64 bytes\n4 × (16 × 16 × 4) = 4096 bytes", 20)
    s.text(54, 354, 407, "The ISA can scale; this kernel is deliberately specialized for 512-bit SVL.", 17, MUTED)
    for i, color in enumerate((TEAL, BLUE, PINK, ORANGE)):
        x, y = 538+(i%2)*171, 148+(i//2)*139
        s.rect(x, y, 155, 123, shade(color), color, radius=6)
        s.text(x+18, y+23, 125, f"ZA{i}.S", 24, color, bold=True)
        s.text(x+18, y+66, 125, "16 × 16\n256 FP32", 16)
    s.band("Four independent tile accumulators let the kernel interleave its updates.")

    s = new("01 / Executing a kernel", "Streaming mode surrounds the complete kernel",
            "C attributes describe the streaming interface and ownership of ZA.",
            "Source: Arm ACLE [1] · src/sme32.c · src/sme_opt.c · src/sme_pack.c",
            "The public computation functions use __arm_locally_streaming with __arm_new(\"za\"). The compiler handles the body entering and leaving streaming mode and creates fresh zero-initialized ZA state according to ACLE. The kernel explicitly resets ZA for each new output tile; subsequent K blocks reload previous C. Internal helpers declare __arm_streaming __arm_inout(\"za\") to share the caller's ZA. This keeps transitions outside the inner tile helper loop. A packing call using ZA has its own function boundary, and total timings include it. The pseudocode omits details of all four tile updates. [1] " + URLS["ACLE"])
    for x, w, heading, body, color in ((48, 190, "Normal caller", "Row-major A, B\nC = A × B", BLUE),
                                       (286, 378, "Streaming function body", "Load Z vectors → FMOPA into ZA\nStore completed tile slices to C", TEAL),
                                       (712, 200, "Return", "Completed C\nCaller continues", BLUE)):
        s.card(x, 147, w, 122, heading, body, color, 15)
    s.arrow(246, 199, 32)
    s.arrow(672, 199, 32)
    s.code(48, 295, 864, '''
        __arm_locally_streaming __arm_new("za")
        void gemm(...) {
            // Tile helpers share ZA: __arm_streaming __arm_inout("za")
            // For each output tile: reset/load ZA, loop over K, store C.
        }
    ''', 14)
    s.band("Zero per new output tile; reload C only when accumulating another K block.")

    s = new("02 / Original implementation", "The 32×32 baseline already uses all four ZA tiles",
            "At each k: load two A vectors, load two B vectors, issue four FMOPA instructions.",
            "Source: src/sme32.c · src/pack.c · src/gemm.h",
            "Original SME32 is our project baseline, not an official standard or a scalar implementation. A0 and A1 contain two adjacent groups of 16 output rows. B0 and B1 contain two groups of 16 output columns. The four outer products form the quadrants of one 32×32 output tile. Both inputs are packed into 16-lane panels before compute. K is blocked into groups of 256; later groups reload the current C tile and continue accumulating. The baseline already uses four ZA tiles, so the optimization cannot be described as merely enabling all tiles.")
    s.text(52, 145, 380, "One 32×32 output tile", 20, bold=True)
    for i, label in enumerate(("A0 × B0", "A0 × B1", "A1 × B0", "A1 × B1")):
        x, y = 60+i%2*151, 188+i//2*103
        color = (TEAL, BLUE, PINK, ORANGE)[i]
        s.rect(x, y, 141, 93, shade(color), radius=5)
        s.text(x+14, y+19, 122, f"ZA{i}.S", 20, color, bold=True)
        s.text(x+14, y+53, 122, label, 17)
    s.card(433, 147, 479, 247, "Original data path", "Row-major A → pack A16\nRow-major B → pack B16\n32×32 microkernel, K block = 256\nRepeated K blocks accumulate into C", TEAL, 20)
    s.band("The baseline has strong arithmetic throughput; packing and C traffic still matter.")

    s = new("02 / Measurement", "Compare the same operation and timing boundary",
            f"M4 Pro · 48 GB · macOS 15.7.3 · Apple Clang 17 · {d.jobs} jobs / {d.samples} samples",
            "Sources: main.c · benchmark.py · examples/m4-pro/environment.json, status.json, verification.csv",
            "All implementations compute C=A×B with FP32 row-major inputs, alpha=1 and beta=0. One calling thread is used; Accelerate BLAS is explicitly forced to single-thread execution and checked. Dimensions are 32,64,128,256,512,1024,2048,4096. Each of three randomized rounds collects seven batch-average samples per case, with calibration targeting at least 30 ms per sample. Inputs are reused and allocation, initialization, process startup, calibration and verification are excluded. Compute mode includes arithmetic and all C traffic but starts with prepared inputs. Accelerate is total-only because its internal packing cannot be exposed separately. Speedups are ratios of within-round medians, summarized across rounds; IQR is not a confidence interval. The run is a steady-state single-calling-thread measurement, not cold-cache latency or chip-wide throughput. Power metadata reported AC Power; the battery line simultaneously reported discharging, so energy-mode or battery claims are not inferred. [4] " + URLS["Threads"])
    s.card(48, 145, 414, 179, "TOTAL / main comparison", "Required packing + full product\nAccelerate receives original inputs\nAllocation and verification excluded", TEAL, 18)
    s.card(482, 145, 430, 179, "COMPUTE / diagnostic", "Inputs already prepared\nAll arithmetic and C traffic included\nSME32 versus optimized SME only", BLUE, 18)
    s.text(56, 348, 400, "Throughput = 2N³ / seconds\nSpeedup = baseline time / backend time", 19, bold=True)
    s.text(490, 348, 414, "3 rounds × 7 batch-average samples\nShading = paired-round IQR, not CI\nAbove 1× means faster than baseline", 17, MUTED)
    s.band("The comparison includes packing and algorithm differences; it does not isolate the ISA.")

    s = new("02 / Baseline results", "Original SME32 trails Accelerate end to end",
            "TOTAL mode: both backends start from the same row-major matrices.",
            notes="Read the left chart as absolute throughput, where higher is better. Read the right chart as Accelerate time divided by original SME32 time, so values below one mean the original is slower. At N=4096, original SME32 delivers about 0.945 TFLOP/s versus 1.571 for Accelerate. The baseline gap is not proof that the SME instructions are slow: the next slide separates prepared-input and total timings. Shading is IQR of three paired-round ratios, not a confidence interval.")
    s.image(42, 128, 876, 307, assets["baseline"])
    s.band(f"At N=4096: {d.value(4096, 'sme32')/1000:.3f} vs {d.value(4096, 'accelerate')/1000:.3f} TFLOP/s, including packing.")

    s = new("03 / Diagnose before tuning", "Small and large matrices reveal different costs",
            "Compute and total are separate measurements; these bars are not additive components.",
            notes="At N=128 the original and optimized compute medians are close, while total timing differs greatly. This points to the data-preparation path as the main opportunity at small sizes. At N=4096 the optimized implementation also improves prepared-input performance substantially, so the large-size result requires changes to the kernel, layout and blocking. Do not subtract these independent medians and call the difference an exact isolated packing measurement. Memory/cache state and separately collected timing variation make that interpretation too strong.")
    s.image(42, 128, 876, 307, assets["diagnosis"])
    s.band("Small N: reduce preparation cost. Large N: improve the kernel and data reuse.")

    s = new("03 / Optimization map", "Use a size-specific path with an explicit fallback",
            "The optimized backend chooses its path before entering the GEMM streaming body.",
            "Source: src/sme_opt.c · src/pack_opt.c · src/gemm.h",
            "The dispatcher requires supported dimensions: multiples of 32, from 32 to 8192. Through N=512 it transposes A using ZA and uses B directly in its original row-major layout, still computing 32×32 tiles. For larger sizes divisible by 64 it uses the 16×64 kernel with A16/B64 packing and larger blocking parameters. Larger dimensions not divisible by 64 use original packing and SME32. This is a transparent fallback without padding or hidden BLAS delegation. The full timing suite contains powers of two, so its larger cases all select the 16×64 path; the non-power-of-two fallback is covered by correctness tests rather than these throughput charts.")
    s.card(48, 151, 264, 226, "N ≤ 512", "Transpose A in ZA\nLeave B row-major\n32×32 output tiles\nNo B packing", PINK, 20)
    s.card(332, 151, 282, 226, "N > 512, N % 64 = 0", "Pack A16 and B64\n16×64 output tiles\nSME2 grouped loads\nCache blocking", TEAL, 20)
    s.card(634, 151, 278, 226, "Other supported N", "Use original packing\nUse original SME32\nNo padding\nNo Accelerate call", BLUE, 20)
    s.text(54, 400, 835, "Supported: square FP32, N a multiple of 32. Benchmark range: N=32…4096.", 17, MUTED)
    s.band("Each optimization changes the cost of delivering the same mathematical product.")

    s = new("03 / Optimization 1", "Transpose A inside ZA to build contiguous panels",
            "Load horizontal slices, read vertical groups, then store packed columns.",
            "Source: src/sme_pack.c · src/pack.c · intrinsic semantics: Arm ACLE [1]",
            "The visual uses a 4×4 toy tile so every value can be followed. The actual packer processes 16×16 FP32 blocks. For each block, 16 row loads populate a ZA tile; groups of four vertical vectors are then read using svread_ver_za32_f32_vg4 and stored contiguously with svst1_f32_x4. The packed A panel places the 16 output-row values for a fixed k next to each other. The baseline uses NEON-based transpose/packing; this path uses SME's tile access to express the transpose directly. No arithmetic GEMM result is stored during packing. [1] " + URLS["ACLE"])
    vals = [[f"{i}{j}" for j in range(4)] for i in range(4)]
    c = [[shade((TEAL, BLUE, PINK, ORANGE)[j], .20) for j in range(4)] for i in range(4)]
    s.text(55, 145, 240, "Input A: row-major", 18, bold=True)
    matrix(s, 61, 188, vals, c, cell=41, size=16)
    s.arrow(249, 258, 66)
    s.text(342, 145, 245, "ZA: read columns", 18, bold=True)
    matrix(s, 351, 188, vals, c, cell=41, size=16)
    s.arrow(540, 258, 59)
    s.text(623, 145, 281, "Packed A: k-major panels", 18, bold=True)
    for k in range(4):
        matrix(s, 631, 188+k*41, [[vals[i][k] for i in range(4)]],
               [[shade((TEAL, BLUE, PINK, ORANGE)[k], .20)]*4], cell=41, size=16)
    s.text(62, 375, 831, "Actual block: 16 horizontal loads → 4 vertical-group reads → 4 grouped stores.", 18)
    s.band("For every k, A[k*16 + lane] now supplies 16 consecutive output-row values.")

    s = new("03 / Optimization 2", "For small matrices, avoid packing B entirely",
            "At N≤512, adjacent B columns are already contiguous in the original row-major input.",
            "Source: src/sme_opt.c: direct_b · src/pack_opt.c: pack_inputs_opt · saved paired speedups",
            "For the small path A still needs packing, now through the ZA transpose. B[k*N+j] is already the beginning of a contiguous group of 32 columns, so a two-vector load supplies B0 and B1 directly to the existing four-FMOPA 32×32 calculation. This eliminates the B preparation pass and keeps a full K sweep for each output tile. The choice is limited to N≤512 because avoiding packing at larger dimensions was not the winning measured strategy. Numbers here compare optimized against original, with total and compute speedups taken separately from speedups.csv. Since packing is excluded in compute, the large total gains paired with near-one compute gains support the small-size diagnosis without pretending to be an isolated packing ablation.")
    s.card(48, 146, 414, 151, "Original", "A → pack A16\nB → pack B16 → compute", TEAL, 21)
    s.card(482, 146, 430, 151, "Small optimized path", "A → transpose in ZA\nB → direct row-major loads → compute", PINK, 21)
    for i, n in enumerate((64, 128, 512)):
        x = 54+i*287
        s.text(x, 322, 260, f"N = {n}", 17, MUTED, bold=True)
        s.text(x, 352, 260, f"{d.ratio(n, 'sme32'):.2f}× total", 29, PINK, bold=True)
        s.text(x, 396, 260, f"{d.ratio(n, 'sme32', 'compute'):.2f}× compute only", 17, MUTED)
    s.band("The small-size gain comes mostly from preparing less data, more efficiently.")

    s = new("03 / Optimization 3", "Reshape the large-matrix kernel to 16×64",
            "One A vector feeds four FMOPA instructions; B arrives as a four-vector group.",
            "Source: src/sme_opt.c: tile_16x64 · src/pack_opt.c · FP32/SVL=512 arithmetic",
            "All four logical 16×16 ZA tiles are arranged across one 16×64 output tile. One 16-lane A vector is reused with each of four consecutive B vectors loaded by svld1_f32_x4. Both shapes perform four FMOPA operations and 2048 nominal FLOPs per k. The new shape actually reads five logical operand vectors per k, versus four in the original 32×32 kernel: 320 versus 256 bytes, ignoring C traffic and cache reuse. Its benefit must therefore be evaluated jointly with grouped loads, panel layout, scheduling and blocking; it is not a reduction in raw operand bytes. The wider logical shape is a design for this implementation, not an architectural requirement.")
    s.text(52, 139, 171, "A: one vector\n16 rows", 18, TEAL, bold=True)
    s.rect(54, 190, 73, 163, shade(TEAL), TEAL)
    s.text(70, 247, 49, "A", 31, TEAL, bold=True)
    s.arrow(145, 263, 61)
    s.text(234, 147, 664, "B: four vectors / 64 columns", 19, BLUE, bold=True)
    for i, color in enumerate((TEAL, BLUE, PINK, ORANGE)):
        x = 232+i*167
        s.rect(x, 187, 156, 38, shade(BLUE))
        s.text(x+14, 196, 130, f"B{i} · 16 lanes", 15, BLUE)
        s.rect(x, 239, 156, 114, shade(color), radius=5)
        s.text(x+17, 259, 131, f"ZA{i}.S", 23, color, bold=True)
        s.text(x+17, 303, 131, f"A × B{i}", 18)
    s.text(55, 382, 842, "Both shapes: 2048 FLOPs/k. Operand loads: 256 B/k (32×32) vs 320 B/k (16×64).", 17)
    s.band("The gain comes from the complete load/layout/blocking design, not fewer operand bytes.")

    s = new("03 / Optimization 4", "Keep the loaded tuple local to each K iteration",
            "Generated instructions matter: inspect the compiler output as well as the C source.",
            "Source: src/sme_opt.c: tile_16x64 comment and loop · build.py · build/sme_opt.disasm.txt",
            "During tuning with Apple Clang 17, carrying the four-vector B tuple across K iterations caused five additional streaming-vector register copies per k. This observation is recorded in the current source comment; the earlier experimental disassembly and paired measurements were not retained in the final repository. Therefore this slide gives no invented speedup or counter measurement for that rejected variant. The final loop loads A and a local B tuple, consumes them immediately, and is unrolled by four. Build auditing verifies FMOPA uses four tiles and that grouped vector loads are present. Those structural checks do not constitute a general proof that any compiler emits zero register moves.")
    s.code(48, 145, 575, '''
        #pragma clang loop unroll_count(4)
        for (size_t k = 0; k < kc; ++k) {
          svfloat32_t va = svld1_f32(pg, a+k*16);
          svfloat32x4_t vb = svld1_f32_x4(pg4, b+k*64);
          svmopa_za32_f32_m(0, pg, pg, va, svget4_f32(vb,0));
          svmopa_za32_f32_m(1, pg, pg, va, svget4_f32(vb,1));
          svmopa_za32_f32_m(2, pg, pg, va, svget4_f32(vb,2));
          svmopa_za32_f32_m(3, pg, pg, va, svget4_f32(vb,3));
        }
    ''', 11.7, 253)
    s.card(645, 145, 267, 253, "Final loop", "Load tuple\nConsume four B vectors\nEnd tuple lifetime\nRepeat, unrolled ×4", PINK, 20)
    s.text(53, 412, 849, "Tuning observation: a tuple carried across iterations generated 5 extra SSVE copies/k.", 15, MUTED)
    s.band("Keep live ranges short and check the actual disassembly on the compiler you use.")

    s = new("03 / Optimization 5", "Use larger K blocks and bounded M/N traversal",
            "Final compute blocks: KC=2048, MC=256, NC=512. Microkernel: 16×64.",
            "Source: src/gemm.h · src/sme_opt.c: optimized · block sizes are measured tuning choices",
            "The actual nesting is pc (K), jc (N), ic (M), j (64 columns), i (16 rows). For a full macroblock, A spans 256×2048 floats = 2 MiB; B spans 2048×512 floats = 4 MiB. These are logical panel footprints, not measured cache residency, and additional C and buffers exist. At N=4096 the original K=256 scheme makes sixteen K passes per output tile, whereas K=2048 needs two. The first pass starts from zero and later passes reload C. This changes logical C load/store frequency: sixteen stores/fifteen reloads versus two stores/one reload per element across the complete product. It does not imply an eightfold reduction in DRAM traffic or an eightfold runtime gain. Blocking values were selected empirically for this M4 Pro/compiler, not claimed universal.")
    s.card(48, 147, 270, 230, "A panel", "MC × KC\n256 × 2048 FP32\n2 MiB logical footprint\n16-row micro-panels", TEAL, 20)
    s.card(337, 147, 280, 230, "B panel", "KC × NC\n2048 × 512 FP32\n4 MiB logical footprint\n64-column micro-panels", BLUE, 20)
    s.card(636, 147, 276, 230, "C at N=4096", "Original: 16 K passes\nOptimized: 2 K passes\nFewer partial-C updates\nReload only after pass 1", PINK, 20)
    s.text(55, 405, 848, "Loop order: K block → N block → M block → 64 columns → 16 rows.", 18, MUTED)
    s.band("Larger K blocks reduce partial-C traffic; M/N blocks organize operand reuse.")

    s = new("03 / Optimization 6", "Tile B packing so source and destination stay local",
            "Large B is copied in 32-row × 256-column chunks, then stored as K×64 panels.",
            "Source: src/pack_opt.c: pack_inputs_opt · src/gemm.h",
            "The full B64 packing loop is pc, jc stepping by 256 columns, kk stepping by 32 K rows, k, j stepping by 64, then four 16-float NEON load/store groups. A chunk covers 32×256×4=32768 bytes of logical source values. Four adjacent 64-column panels are filled over those 32 rows before moving on. The four-vector FP32 load in the compute kernel can then fetch one contiguous 64-value B row from a packed micro-panel. These NEON packing groups should not be confused with the SME2 four-vector loads in the streaming compute function. Packing remains part of total timing. The chosen packing and compute blocking constants are different: 256 packing columns versus NC=512 compute columns.")
    s.text(53, 148, 400, "Source: original row-major B", 19, bold=True)
    for r in range(4):
        for c in range(4):
            s.rect(60+c*83, 195+r*43, 78, 38, shade((TEAL, BLUE, PINK, ORANGE)[c], .17))
    s.text(63, 384, 378, "A chunk = 32 K rows × 256 columns\n32 KiB of logical FP32 source values", 17, MUTED)
    s.arrow(431, 270, 69)
    s.text(536, 148, 370, "Destination: four K×64 panels", 19, bold=True)
    for c in range(4):
        color = (TEAL, BLUE, PINK, ORANGE)[c]
        s.rect(544+c*89, 193, 77, 164, shade(color), color, radius=5)
        s.text(553+c*89, 219, 64, f"B{c}", 22, color, bold=True)
        s.text(553+c*89, 271, 64, "K\n×\n64", 17)
    s.text(542, 384, 362, "Compute loads 64 consecutive floats\nper k using one vector-group load.", 17, MUTED)
    s.band("Do not trade a faster compute loop for expensive scattered writes in the packer.")

    s = new("04 / Final throughput", "The optimized version closes most of the library gap",
            "TOTAL mode, including required packing. Higher throughput is better.",
            notes="All three curves are generated directly from summary.csv. Each throughput value is 2*N^3 divided by the pooled median time, expressed in TFLOP/s. At N=512 the optimized and Accelerate curves are almost identical. At N=2048 the optimized implementation remains slower. At N=4096 both are around 1.57 TFLOP/s, while original SME32 is about 0.945. Do not claim the optimized version universally beats Accelerate, or that these results show all-chip peak throughput.")
    s.image(39, 128, 666, 311, assets["throughput"])
    s.card(725, 150, 187, 130, "N = 512", f"Optimized\n{d.value(512, 'sme_opt')/1000:.2f} TFLOP/s", PINK, 18)
    s.card(725, 299, 187, 130, "N = 4096", f"Optimized\n{d.value(4096, 'sme_opt')/1000:.2f} TFLOP/s", PINK, 18)
    s.band("Near Accelerate at many sizes; N=2048 remains a visible gap.")

    s = new("04 / Final speedup", "Against Accelerate: small-size wins, large-size parity",
            "Speedup = Accelerate time / backend time. Above 1× means faster than Accelerate.",
            notes="Ratios come from speedups.csv: medians of within-round timing ratios, with IQR across three rounds. This is why they can differ slightly from ratios of the throughput values or pooled medians. At N=32 the paired optimized speedup is about 2.69×, but absolute batch-average calls are sub-microsecond, so do not generalize this to tail latency. At 4096 the optimized/Accelerate ratio is about 1.006× with IQR crossing 1, which is effectively parity for interpretation. At 2048 it is 0.946×, equivalent to about 5.7% longer runtime because inverse speedup minus one is the slowdown. A few percent elsewhere should be read with variability in mind.")
    s.image(39, 128, 666, 311, assets["speedup"])
    s.card(725, 144, 187, 130, "N = 32", f"{d.ratio(32):.2f}×\nsmall-call benefit", PINK, 20)
    s.card(725, 291, 187, 138, "N = 2048", f"{d.ratio(2048):.3f}×\n≈{(1/d.ratio(2048)-1)*100:.0f}% longer time", ORANGE, 20)
    s.band("Shading = IQR of paired-round speedups. It is not a confidence interval.")

    s = new("04 / Improvement over the original", "The optimized path improves preparation and compute",
            "Same two SME backends; total and prepared-input timings answer different questions.",
            notes="Pink includes packing and blue excludes it. At N=128 total improves by about 2.40× while compute improves only about 1.03×, consistent with preparation being the main small-size opportunity. At N=4096 total improves by about 1.67× and compute by about 1.64×, consistent with the large-path design also improving arithmetic/data delivery. These before/after comparisons combine several changes; no per-optimization causal speedup or ablation is claimed. The K-block count and logical panel sizes explain plausible mechanisms, while the measured curves are the end-to-end evidence.")
    s.image(42, 128, 876, 307, assets["improvement"])
    s.band(f"At N=4096: {d.ratio(4096, 'sme32'):.2f}× total and {d.ratio(4096, 'sme32', 'compute'):.2f}× compute over original SME32.")

    s = new("04 / Selected measured results", "Read speedup together with absolute throughput",
            "TOTAL mode · GFLOP/s from pooled median times · speedups from paired rounds",
            notes="This table is derived automatically, not typed from rounded values. The selected rows cover the preparation-dominated region, the dispatch threshold and all measured large sizes. Original and optimized columns are GFLOP/s, while the final columns are dimensionless paired speedups. For the 2048 case Accelerate is faster; at 4096 the library and custom kernel are effectively tied. Source data remain available with hashes in data_manifest.json and the original complete run. The full-range charts include N=32,64 and256 as well.")
    headers = ["N", "Original SME32", "Optimized SME", "Accelerate", "Opt / original", "Opt / Accelerate"]
    xs, widths = [49, 120, 288, 462, 621, 767], [65, 161, 166, 151, 141, 145]
    for x, w, text in zip(xs, widths, headers):
        s.text(x+7, 153, w-10, text, 13, MUTED, bold=True)
    for i, n in enumerate((128, 512, 1024, 2048, 4096)):
        y = 197+i*44
        s.rect(48, y-5, 864, 41, WHITE if i%2 == 0 else "#EDF1F6", radius=3)
        vals = [str(n), f"{d.value(n, 'sme32'):.0f}", f"{d.value(n, 'sme_opt'):.0f}",
                f"{d.value(n, 'accelerate'):.0f}", f"{d.ratio(n, 'sme32'):.2f}×", f"{d.ratio(n):.2f}×"]
        for j, (x, w, value) in enumerate(zip(xs, widths, vals)):
            s.text(x+7, y+4, w-10, value, 20, PINK if j in (2, 4) else ORANGE if j==3 else INK, bold=j==2)
    s.band("No universal winner: performance depends on N, packing, cache behavior, and compiler output.")

    s = new("05 / Reproduce and extend", "Keep correctness, generated code, and timing together",
            "The saved example is complete. A new run creates fresh data; slides never run the benchmark.",
            "Sources: README.md · check.py · build.py · benchmark.py · plot.py · slides/build_slides.py",
            "The repository checks packing and GEMM with AddressSanitizer and UndefinedBehaviorSanitizer, a scalar double reference, multiple sizes/seeds and output guards. Benchmark jobs perform numerical checks before and after sampling, including a finiteness scan. Build manifests record source and binary hashes, compiler flags and disassembly properties. The runner rejects a source/binary mismatch. Limitations: FP32 only, square multiples of32 only, warm reused inputs, one calling thread without guaranteed CPU affinity, no energy or multicore results, and no isolated SME-versus-NEON experiment. To study pure ISA uplift one would need a controlled no-SME implementation and comparable packing, blocking, compiler and arithmetic semantics. The current result is useful as an implementation comparison. Slides can be rebuilt from saved results without running GEMM.")
    s.code(48, 144, 540, '''
        python3 build.py
        python3 check.py
        python3 benchmark.py --suite full --output results/run1
        python3 plot.py results/run1
    ''', 12.5, 119)
    s.code(48, 281, 540, '''
        # Rebuild this deck from existing results only:
        python3 slides/build_slides.py
    ''', 12.5, 78)
    s.card(610, 144, 302, 264, "Scope of the evidence", "Square FP32, one thread\nReused input buffers\nOne machine / compiler\nNo isolated ISA speedup\nNo per-change ablation", BLUE, 18)
    s.text(54, 381, 525, "Next experiments: independent runs, other sizes,\ncontrolled NEON baseline, and multicore scaling.", 16, MUTED)
    s.band("Fast GEMM comes from arithmetic, operand delivery, and measurement discipline together.")

    s = new("Appendix / References and reading map", "Follow the explanation back to its sources",
            "Primary architecture/library references and the exact local implementation.",
            "External references checked 13 November 2026 · see speaker_notes.md for per-slide details",
            "[1] Arm C Language Extensions: SME language extensions and intrinsics, streaming/ZA attributes. " + URLS["ACLE"] + "\n[2] Friedrich Schiller University Jena, Hello SME microbenchmarks: streaming vector length, FMOPA, tile reuse. This external source provides architectural context, not the performance dataset for our charts. " + URLS["Jena"] + "\n[3] Apple Accelerate overview: BLAS/LAPACK and processor-dependent implementation selection. " + URLS["Apple"] + "\n[4] Apple BLAS_THREADING_SINGLE_THREADED. " + URLS["Threads"] + "\nLocal sources: src/sme32.c, src/sme_opt.c, src/sme_pack.c, src/pack_opt.c, src/gemm.h, main.c, benchmark.py. All figures showing performance use examples/m4-pro/summary.csv and speedups.csv, checked against raw.jsonl. data_manifest.json records input hashes.")
    refs = [("[1] Arm C Language Extensions", "Streaming interfaces, ZA state, SME/SME2 intrinsics", URLS["ACLE"]),
            ("[2] Hello SME / University Jena", "FMOPA, streaming vector length, tile reuse", URLS["Jena"]),
            ("[3] Apple Accelerate", "BLAS/LAPACK and runtime processor selection", URLS["Apple"]),
            ("[4] Apple BLAS threading", "Single-thread execution control", URLS["Threads"])]
    for i, (title, desc, url) in enumerate(refs):
        y = 145+i*67
        s.text(53, y, 435, title, 18, TEAL, bold=True, link=url)
        s.text(53, y+27, 439, desc, 14, MUTED)
    s.card(524, 145, 388, 261, "Local reading order", "src/sme32.c → original kernel\nsrc/sme_pack.c → A transpose\nsrc/pack_opt.c → dispatch + B packing\nsrc/sme_opt.c → optimized kernels\nexamples/m4-pro/ → all measured data\nspeaker_notes.md → detailed explanations", BLUE, 15.5)
    s.band("Architecture diagrams are explanatory; every performance curve uses local measured results.")
    return deck


def export(deck, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Pt(W), Pt(H)
    prs.core_properties.title = "From outer products to competitive GEMM"
    prs.core_properties.subject = "Arm SME / SME2 on Apple M4 Pro"
    prs.core_properties.author = "apple-sme-gemm"
    pdf = canvas.Canvas(str(out/"sme-on-apple-m4-pro.pdf"), pagesize=(W, H))
    pdf.setTitle(prs.core_properties.title)
    pdf.setAuthor("apple-sme-gemm")
    qa = []
    for number, s in enumerate(deck, 1):
        s.text(880, 504, 32, f"{number:02d}", 9, MUTED)
        pslide = prs.slides.add_slide(prs.slide_layouts[6])
        pslide.notes_slide.notes_text_frame.text = s.notes
        for item in s.items:
            k = item["kind"]
            x, y = item["x"], item["y"]
            if k in ("rect", "text", "image", "arrow"):
                w, h = item["w"], item["h"]
                if x < 0 or y < 0 or x+w > W+1 or y+h > H+1:
                    qa.append(f"Slide {number}: {k} outside slide bounds: {item}")
            if k == "rect":
                shape = pslide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if item["radius"] else MSO_SHAPE.RECTANGLE,
                                               Pt(x), Pt(y), Pt(w), Pt(h))
                if item["radius"]:
                    shape.adjustments[0] = min(.15, item["radius"]/min(w, h))
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(item["fill"][1:])
                if item["stroke"]:
                    shape.line.color.rgb = RGBColor.from_string(item["stroke"][1:])
                else:
                    shape.line.fill.background()
                pdf.setFillColor(item["fill"])
                pdf.setStrokeColor(item["stroke"] or item["fill"])
                pdf.roundRect(x, H-y-h, w, h, item["radius"], stroke=bool(item["stroke"]), fill=1)
            elif k == "line":
                shape = pslide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x), Pt(y), Pt(item["x2"]), Pt(item["y2"]))
                shape.line.color.rgb = RGBColor.from_string(item["color"][1:])
                shape.line.width = Pt(item["width"])
                pdf.setStrokeColor(item["color"])
                pdf.setLineWidth(item["width"])
                pdf.line(x, H-y, item["x2"], H-item["y2"])
            elif k == "text":
                box = pslide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
                tf = box.text_frame
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.word_wrap = False
                for i, line in enumerate(item["lines"]):
                    if pdfmetrics.stringWidth(line, item["font"], item["size"]) > w:
                        qa.append(f"Slide {number}: text wider than box: {line}")
                    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    para.space_before = para.space_after = Pt(0)
                    para.line_spacing = Pt(item["size"]*1.24)
                    run = para.add_run()
                    run.text = line
                    run.font.name = "Courier New" if item["font"] == "Mono" else "Arial"
                    run.font.size = Pt(item["size"])
                    run.font.bold = item["font"] == "Bold"
                    run.font.color.rgb = RGBColor.from_string(item["color"][1:])
                    if item["link"]:
                        run.hyperlink.address = item["link"]
                    pdf.setFillColor(item["color"])
                    pdf.setFont(item["font"], item["size"])
                    pdf.drawString(x, H-y-item["size"]*.94-i*item["size"]*1.24, line)
                if item["link"]:
                    pdf.linkURL(item["link"], (x, H-y-h, x+w, H-y), relative=0)
            elif k == "image":
                pslide.shapes.add_picture(item["path"], Pt(x), Pt(y), Pt(w), Pt(h))
                pdf.drawImage(item["path"], x, H-y-h, w, h, mask="auto")
            elif k == "arrow":
                shape = pslide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Pt(x), Pt(y), Pt(w), Pt(h))
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(item["color"][1:])
                shape.line.fill.background()
                pdf.setFillColor(item["color"])
                path = pdf.beginPath()
                points = [(x, H-y-4), (x+w-10, H-y-4), (x+w-10, H-y),
                          (x+w, H-y-6.5), (x+w-10, H-y-13), (x+w-10, H-y-9), (x, H-y-9)]
                path.moveTo(*points[0])
                for point in points[1:]:
                    path.lineTo(*point)
                path.close()
                pdf.drawPath(path, fill=1, stroke=0)
        pdf.showPage()
        # Text-to-text collisions are unintended; backgrounds intentionally overlap text.
        texts = [t for t in s.items if t["kind"] == "text"]
        for i, a in enumerate(texts):
            for b in texts[i+1:]:
                iw = min(a["x"]+a["w"], b["x"]+b["w"])-max(a["x"], b["x"])
                ah = a["size"]*1.24*len(a["lines"])
                bh = b["size"]*1.24*len(b["lines"])
                ih = min(a["y"]+ah, b["y"]+bh)-max(a["y"], b["y"])
                if iw > 3 and ih > 3:
                    qa.append(f"Slide {number}: overlapping text: {a['lines']} / {b['lines']}")
    pdf.save()
    prs.save(out/"sme-on-apple-m4-pro.pptx")
    if qa:
        raise ValueError("Layout checks failed:\n" + "\n".join(qa))
    notes = "# Speaker notes\n\nThe slide titles, diagrams, and chart assets are generated by `build_slides.py`.\nAll performance figures use the saved run; no new benchmark is executed.\n\n"
    for i, s in enumerate(deck, 1):
        notes += f"## {i:02d}. {s.title}\n\n{s.notes}\n\n"
    (out/"speaker_notes.md").write_text(notes.rstrip()+"\n")
    (out/"slide_outline.json").write_text(json.dumps([
        {"number": i, "title": s.title, "text": ["\n".join(t["lines"]) for t in s.items if t["kind"] == "text"]}
        for i, s in enumerate(deck, 1)], indent=2)+"\n")


def preview(out, count):
    """Optional QA render; PyMuPDF is not needed to build PPTX/PDF."""
    import pymupdf
    doc = pymupdf.open(out/"sme-on-apple-m4-pro.pdf")
    if len(doc) != count:
        raise ValueError("PDF page count does not match the PowerPoint")
    target = out/"previews"
    target.mkdir(exist_ok=True)
    thumb_w, thumb_h, gap = 480, 270, 18
    cols, rows = 3, math.ceil(count/3)
    sheet = Image.new("RGB", (cols*(thumb_w+gap)+gap, rows*(thumb_h+35+gap)+gap), "#E5E9EF")
    draw = ImageDraw.Draw(sheet)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(target/f"slide-{i+1:02d}.png")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = gap+(i%cols)*(thumb_w+gap), gap+(i//cols)*(thumb_h+35+gap)
        sheet.paste(img, (x, y))
        draw.text((x, y+thumb_h+8), f"Slide {i+1:02d}", fill=INK)
    sheet.save(out/"overview.png")
    doc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, help="Full result directory; default: <repo>/examples/m4-pro")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--preview", action="store_true", help="Also render previews; requires PyMuPDF")
    args = parser.parse_args()
    register_fonts()
    repo, out = args.repo.resolve(), args.output.resolve()
    result = (args.results or repo/"examples/m4-pro").resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = Data(repo, result)
    assets = charts(data, out/"assets")
    deck = make_slides(data, assets)
    export(deck, out)
    paths = ["summary.csv", "speedups.csv", "raw.jsonl", "environment.json", "status.json", "verification.csv"]
    manifest = {
        "data_source": str(result.relative_to(repo)) if result.is_relative_to(repo) else str(result),
        "input_sha256": {name: sha(result/name) for name in paths},
        "source_sha256": data.env["build"]["source_sha256"],
        "slides": len(deck), "jobs": data.jobs, "samples": data.samples,
        "benchmark_started_at": data.env["started_at"],
        "charts": "Throughput from pooled timing medians; speedups from paired-round medians; shading is IQR, not CI.",
        "checks": ["Complete full suite", "Throughput formula", "Speedup medians and IQR against raw jobs",
                   "Measured sources match current sources", "Text widths, slide boundaries and text collisions"],
    }
    (out/"data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.preview:
        preview(out, len(deck))
    print(f"Created {len(deck)} slides: PPTX + PDF + notes + chart assets")
    print(f"Validated {data.jobs} jobs and {data.samples} samples. No benchmark was run.")


if __name__ == "__main__":
    main()
