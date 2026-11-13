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
import statistics
import sys
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/apple-sme-slide-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/apple-sme-slide-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COLORS = {"sme": PINK, "accelerate": ORANGE}
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
        self.env=json.loads((result/'environment.json').read_text())
        status=json.loads((result/'status.json').read_text())
        if not status['complete']: raise ValueError('A complete benchmark run is required')
        summary=load_csv(result/'summary.csv')
        ratios=load_csv(result/'speedups.csv')
        self.index={(int(r['size']),r['backend']):r for r in summary}
        self.ratios={int(r['size']):r for r in ratios}
        sizes=sorted({int(r['size']) for r in summary})
        if sizes != [32,64,128,256,512,1024,2048,4096]:
            raise ValueError('This deck requires the full eight-size suite')
        if {r['backend'] for r in summary} != {'sme','accelerate'}:
            raise ValueError('Expected exactly SME and Accelerate')
        records=[json.loads(line) for line in (result/'raw.jsonl').read_text().splitlines()]
        self.jobs=len(records)
        self.samples=sum(len(r['samples_ms']) for r in records)
        paired={(r['size'],r['round'],r['backend']):statistics.median(r['samples_ms']) for r in records}
        for n in sizes:
            for backend in COLORS:
                row=self.index[n,backend]
                if not math.isclose(2*n**3/float(row['median_ms'])/1e6,float(row['gflops']),rel_tol=1e-10):
                    raise ValueError('Throughput does not match timings')
            row=self.ratios[n]
            values=[paired[n,k,'accelerate']/paired[n,k,'sme'] for k in range(1,int(row['paired_rounds'])+1)]
            for field,val in [('speedup_median',statistics.median(values)),('speedup_p25',quantile(values,.25)),('speedup_p75',quantile(values,.75))]:
                if not math.isclose(float(row[field]),val,rel_tol=1e-10): raise ValueError('Speedup/IQR does not match raw jobs')
        for name,digest in self.env['build']['source_sha256'].items():
            if sha(repo/name)!=digest: raise ValueError('Source changed since the measured build: '+name)

    def value(self,n,backend):
        return float(self.index[n,backend]['gflops'])

    def ratio(self,n):
        return float(self.ratios[n]['speedup_median'])


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


def make_slides(d, assets):
    deck=[]
    source='Measured: examples/m4-pro/ · square FP32 · one thread · packing included'
    def new(section,title,subtitle='',citation=source,notes=''):
        s=Slide(section,title,subtitle,citation,notes);deck.append(s);return s

    s=new('Apple SME GEMM','SME and Accelerate on Apple M4 Pro',
          'One complete SME implementation and one vendor-library reference.',
          notes='This deck explains the implementation currently present in the repository. The SME backend uses size-dependent kernels as one implementation: direct B through N=512 and a blocked 16×64 kernel above it. Accelerate is the other backend. All results include preparation of the inputs. There is no historical baseline implementation in the worktree.')
    s.text(52,158,535,'Understand the matrix operation.\nDeliver its operands efficiently.\nMeasure the complete product.',30,bold=True)
    s.card(650,150,262,252,'What this deck covers','Outer products and ZA\nFinal packing and kernels\nBlocking and edge handling\nThroughput and speedup\nReproducible measurements',PINK,19)
    s.text(54,341,540,'Native C intrinsics · FP32 · square matrices\nSME2 · 512-bit streaming vector length',19,MUTED)
    s.band('Architecture diagrams explain the design; the charts show measured local results.')

    s=new('01 / Concepts','SME is an instruction set; Accelerate is a library',
          'SME means Scalable Matrix Extension.',
          'Sources: Arm ACLE [1] · Apple Accelerate [3] · src/accelerate.c',
          'SME is an Arm architectural extension exposed through instructions and C intrinsics. SME2 adds grouped-vector operations used in this implementation. Accelerate is Apple’s framework, including BLAS and LAPACK. Its cblas_sgemm entry point selects an internal implementation appropriate to the CPU. This experiment does not inspect that library’s internal instruction path, so it cannot isolate the effect of using SME versus disabling it. [1] '+URLS['ACLE']+' [3] '+URLS['Apple'])
    s.card(48,147,414,229,'Arm SME / SME2','Streaming execution mode\nZ vectors and predicates\nZA matrix accumulators\nSME2 vector-group loads',TEAL,20)
    s.card(482,147,430,229,'Apple Accelerate','BLAS interface: cblas_sgemm\nComputes C = A × B\nLibrary selects its internal path\nForced to one thread here',ORANGE,20)
    s.band('This is an implementation comparison, with the same inputs and timing boundary.')

    s=new('01 / Matrix arithmetic','One outer product updates many output values',
          'The full product is a sum over K: C += A[:, k] × B[k, :].',
          'Sources: Arm ACLE [1] · Hello SME [2] · teaching example constructed for this deck',
          'For one value of k, take a column of A and a row of B. Their outer product adds a rectangular update to C. Repeat for every k. FMOPA performs the predicated outer-product accumulate into a ZA tile. Our numerical toy example uses a two-value column and a three-value row; real FP32 vectors here have 16 lanes. Each FMOPA updates 256 values with fused multiply-add operations, or 512 nominal FLOPs. [1] '+URLS['ACLE']+' [2] '+URLS['Jena'])
    s.text(70,147,170,'A[:, k]',19,TEAL,bold=True)
    matrix(s,95,202,[[2],[3]],cell=58,size=23)
    s.text(179,235,45,'×',30,MUTED)
    s.text(248,147,240,'B[k, :]',19,BLUE,bold=True)
    matrix(s,245,230,[[4,5,6]],[[shade(BLUE)]*3],cell=58,size=23)
    s.arrow(450,253,56)
    s.text(548,147,320,'Update added to C',19,PINK,bold=True)
    matrix(s,560,202,[[8,10,12],[12,15,18]],[[shade(PINK)]*3]*2,cell=65,size=23)
    s.text(60,370,825,'16 FP32 lanes × 16 lanes = 256 updates per FMOPA on this machine.',20)
    s.band('256 fused multiply-adds = 512 nominal FLOPs per FP32 FMOPA.')

    s=new('01 / Register storage','ZA exposes four logical FP32 tiles at 512-bit SVL',
          'The runtime checks for 16 FP32 lanes before executing the SME backend.',
          'Sources: Arm ACLE [1] · Hello SME [2] · src/sme.c · main.c',
          'At a 512-bit streaming vector length, a Z vector holds 16 FP32 values. ZA contains 4096 bytes and its FP32 view provides four 16×16 tiles. The diagram shows logical tile views rather than physical SRAM geometry. The implementation is specialized to this SVL and rejects others. Public SME computation functions use locally-streaming/private-ZA attributes; internal tile helpers share the caller’s ZA, keeping mode transitions outside the inner tile loop. Explicit zeroing starts a new output tile; subsequent K blocks reload C. [1] '+URLS['ACLE']+' [2] '+URLS['Jena'])
    s.text(53,145,420,'A Z vector: 16 × 32-bit values',20,bold=True)
    for i in range(16): s.rect(54+i*25,191,23,35,shade(TEAL,.12 if i%2 else .23))
    s.text(54,250,410,'One ZA tile: 16 × 16 FP32\nFour tiles: 4 × 1024 bytes\nTotal ZA storage: 4096 bytes',20)
    s.text(54,355,405,'Predicates select active lanes.\nThe size contract uses multiples of 32.',17,MUTED)
    for i,color in enumerate((TEAL,BLUE,PINK,ORANGE)):
        x,y=538+(i%2)*171,148+(i//2)*139
        s.rect(x,y,155,123,shade(color),color,radius=6)
        s.text(x+18,y+23,125,f'ZA{i}.S',24,color,bold=True)
        s.text(x+18,y+66,125,'16 × 16\n256 FP32',16)
    s.band('Both kernels interleave updates across all four logical ZA tile accumulators.')

    s=new('02 / Final implementation','One SME backend adapts its data path to matrix size',
          'The two kernels are complementary parts of the final implementation.',
          'Sources: src/gemm.h · src/sme.c · src/pack.c',
          'For N≤512, the implementation transposes A into 16-row panels and loads B directly in its original row-major form; a 32×32 output tile uses all four ZA tiles. For larger matrices it packs B into 64-column panels and uses a blocked 16×64 kernel. This applies to every supported dimension, including a final 32-column tail. There is no call to a separate legacy backend and no delegation to Accelerate. Supported dimensions are multiples of32 from32 through8192. These size-dependent choices are parts of the same final implementation, not selectable experimental versions.')
    s.card(48,151,414,238,'N ≤ 512','A: transpose in ZA\nB: original row-major input\nOutput tile: 32×32\nNo B preparation buffer',PINK,21)
    s.card(482,151,430,238,'N > 512','A16 / B64 packed panels\nOutput tile: 16×64\nKC=2048, MC=256, NC=512\nFinal 32 columns handled in place',TEAL,21)
    s.band('Input and output matrices remain N×N row-major FP32 arrays.')

    s=new('02 / Preparing A','Build contiguous A panels by reading ZA columns',
          'Horizontal tile loads followed by vertical-group reads express the transpose.',
          'Source: src/pack.c · intrinsic semantics: Arm ACLE [1]',
          'The diagram labels a 4×4 toy tile by row and column. The actual packer loads16 horizontal slices of a16×16 FP32 block into ZA, then reads columns in groups of4 using svread_ver_za32_f32_vg4 and writes them contiguously. The packed panel puts16 A values for one k next to one another, exactly as the FMOPA operand requires. This preparation is timed on every SME call. [1] '+URLS['ACLE'])
    vals=[[f'{i}{j}' for j in range(4)] for i in range(4)]
    colors=[[shade((TEAL,BLUE,PINK,ORANGE)[j],.20) for j in range(4)] for i in range(4)]
    s.text(55,145,240,'Input: row-major A',18,bold=True)
    matrix(s,61,188,vals,colors,cell=41,size=16)
    s.arrow(249,258,66)
    s.text(342,145,245,'ZA: read columns',18,bold=True)
    matrix(s,351,188,vals,colors,cell=41,size=16)
    s.arrow(540,258,59)
    s.text(623,145,281,'Packed A: k-major panels',18,bold=True)
    for k in range(4):
        matrix(s,631,188+k*41,[[vals[i][k] for i in range(4)]],[[shade((TEAL,BLUE,PINK,ORANGE)[k],.20)]*4],cell=41,size=16)
    s.text(62,375,831,'Actual 16×16 block: 16 row loads → 4 column-group reads → 4 grouped stores.',18)
    s.band('A[k*16 + lane] supplies 16 consecutive output-row values for each k.')

    s=new('02 / Small matrices','For N≤512, B can be consumed without packing',
          'Each row already contains the consecutive columns needed by the kernel.',
          'Source: src/sme.c: direct_b · src/pack.c: pack_inputs',
          'A0 and A1 represent two groups of16 output rows. A two-vector B load supplies B0 and B1, representing32 consecutive output columns. Four outer products fill the quadrants of a32×32 output tile. A is prepared with the ZA transpose; B requires no packing or extra buffer. This saves work at sizes where copying B is not beneficial. The small path traverses all K values before storing the completed tile.')
    s.text(52,145,380,'One 32×32 output tile',20,bold=True)
    for i,label in enumerate(('A0 × B0','A0 × B1','A1 × B0','A1 × B1')):
        x,y=60+i%2*151,188+i//2*103;color=(TEAL,BLUE,PINK,ORANGE)[i]
        s.rect(x,y,141,93,shade(color),radius=5)
        s.text(x+14,y+19,122,f'ZA{i}.S',20,color,bold=True)
        s.text(x+14,y+53,122,label,17)
    s.card(433,147,479,247,'Per k','Load A0 and A1 from packed A\nLoad B0 and B1 from original B\nIssue four FMOPA instructions\nKeep the tile in ZA until K is complete',PINK,20)
    s.band('The four tile updates perform 2048 nominal FLOPs for each k.')

    s=new('02 / Large matrices','The 16×64 kernel reuses one A vector four times',
          'SME2 loads four contiguous B vectors as a group.',
          'Source: src/sme.c: tile_16x64 · src/pack.c',
          'Four logical16×16 ZA tiles are arranged across a16×64 output region. One A vector is reused with four B vectors from svld1_f32_x4. A tuple is consumed in the same K iteration, keeping its live range short; the loop is unrolled four times. This shape performs2048 nominal FLOPs/k and loads320 logical operand bytes/k, excluding C traffic and cache reuse. Its performance depends on the complete grouping, packing and blocking design; tile shape alone does not guarantee efficiency.')
    s.text(52,139,171,'A: one vector\n16 rows',18,TEAL,bold=True)
    s.rect(54,190,73,163,shade(TEAL),TEAL)
    s.text(70,247,49,'A',31,TEAL,bold=True)
    s.arrow(145,263,61)
    s.text(234,147,664,'B: four vectors / 64 columns',19,BLUE,bold=True)
    for i,color in enumerate((TEAL,BLUE,PINK,ORANGE)):
        x=232+i*167
        s.rect(x,187,156,38,shade(BLUE));s.text(x+14,196,130,f'B{i} · 16 lanes',15,BLUE)
        s.rect(x,239,156,114,shade(color),radius=5)
        s.text(x+17,259,131,f'ZA{i}.S',23,color,bold=True)
        s.text(x+17,303,131,f'A × B{i}',18)
    s.text(55,382,842,'Per k: 4 FMOPA instructions · 2048 FLOPs · 320 operand bytes, excluding C.',18)
    s.band('Grouped loads, short live ranges, and matching packed panels work together.')

    s=new('02 / Compute loop','Load, consume, and retire the tuple in one iteration',
          'The final loop is unrolled four times; helpers share the caller’s ZA state.',
          'Source: src/sme.c: tile_16x64 · generated instructions audited by build.py',
          'The code is the inner arithmetic loop used in the final large-matrix kernel. va is reused by all four outer products. vb is loaded and consumed within the same iteration. The build checks that all four FMOPA tile destinations and grouped vector loads are present in the generated object code. These structural checks complement numerical tests and end-to-end timing; they do not establish peak throughput by themselves.')
    s.code(48,145,575,'''
        #pragma clang loop unroll_count(4)
        for (size_t k = 0; k < kc; ++k) {
          svfloat32_t va = svld1_f32(pg, a+k*16);
          svfloat32x4_t vb = svld1_f32_x4(pg4, b+k*64);
          svmopa_za32_f32_m(0, pg, pg, va, svget4_f32(vb,0));
          svmopa_za32_f32_m(1, pg, pg, va, svget4_f32(vb,1));
          svmopa_za32_f32_m(2, pg, pg, va, svget4_f32(vb,2));
          svmopa_za32_f32_m(3, pg, pg, va, svget4_f32(vb,3));
        }
    ''',11.7,253)
    s.card(645,145,267,253,'Within each iteration','Load A and B\nConsume all B vectors\nReuse A four times\nEnd tuple lifetime',PINK,20)
    s.band('Inspect the generated code on the compiler actually used for the benchmark.')

    s=new('02 / Blocking and edges','Bound the working panels and handle the last 32 columns',
          'Compute blocks: KC=2048, MC=256, NC=512. B packing chunks: 32×256.',
          'Sources: src/gemm.h · src/pack.c · src/sme.c',
          'Compute traversal is K block, N block, M block,64 columns,16 rows. Full macro-panels have logical sizes256×2048×4=2 MiB for A and2048×512×4=4 MiB for B, not guaranteed cache residency. B packing visits32 K rows and256 columns at a time, keeping accesses local. If N is not divisible by64, only the final32-column B panel is extended with32 zero columns. A is not padded; C retains its N stride, and loads/stores for nonexistent output columns are skipped. No other implementation is used. Tests cover every packed value including zeros, output guards, and partial K blocks.')
    s.card(48,147,270,230,'A panel','256 × 2048 FP32\n2 MiB logical footprint\n16-row micro-panels\nNo padding',TEAL,19)
    s.card(337,147,280,230,'B panel','2048 × 512 FP32\n4 MiB logical footprint\n64-column micro-panels\nOnly the final tail is padded',BLUE,19)
    s.card(636,147,276,230,'C output','Original N×N allocation\n16×64 output regions\nSkip nonexistent columns\nReload for later K blocks',PINK,19)
    s.text(55,405,848,'At N=4096, KC=2048 needs two K passes per output region.',18,MUTED)
    s.band('Edge handling belongs to the same final kernel; no legacy fallback is retained.')

    s=new('03 / Measurement','Compare the same complete row-major matrix product',
          f'{d.env["cpu"]} · {d.env["memory"]} · {d.jobs} jobs / {d.samples} samples',
          'Sources: main.c · benchmark.py · examples/m4-pro/environment.json · Apple BLAS threading [4]',
          'Both backends compute C=A×B with FP32 row-major inputs, alpha1 and beta0. SME packing is included on every call. Allocation, initialization, verification, process startup and calibration are excluded. One calling thread uses USER_INITIATED QoS; Accelerate BLAS is explicitly forced to one thread. The full suite has eight sizes, three randomized rounds and seven batch-average samples per backend/size; calibration targets at least30ms per batch. Inputs are reused. Throughput is2N^3 divided by pooled median seconds. Speedup is the median of Accelerate/SME timing ratios paired within each round. IQR shading is not a confidence interval. [4] '+URLS['Threads'])
    s.card(48,147,414,195,'Timed','Required packing on every SME call\nThe complete product C = A × B\nAll C loads and stores\nAccelerate’s public SGEMM call',TEAL,19)
    s.card(482,147,430,195,'Protocol','One thread for both backends\nIdentical deterministic inputs\n3 rounds × 7 batch-average samples\nCases and backends randomized',BLUE,19)
    s.text(56,365,830,'Higher throughput is better. Speedup above 1× means SME is faster than Accelerate.',18)
    s.band('Warm reused inputs, one calling thread: these measurements are not chip-wide peak throughput.')

    s=new('03 / Measured comparison','SME versus Accelerate: throughput and speedup',
          'All required packing is included. Speedup = Accelerate time / SME time.',
          notes='The chart is generated by the same plot.py function used for ordinary benchmark runs. Both series come from the current full run. Throughput uses the pooled median timing; paired speedups use per-round median ratios, so they need not equal a ratio of pooled medians. Interpret small differences using the IQR rather than declaring a universal winner. This compares two implementations and does not inspect Accelerate’s internal instruction path.')
    s.image(43,125,874,318,assets['comparison'])
    s.band('Shading shows variability across paired rounds; absolute throughput remains essential context.')

    s=new('03 / Selected results','Read the ratios alongside absolute performance',
          'Packing included · GFLOP/s from pooled medians · speedup from paired rounds',
          notes='The table is generated from summary.csv and speedups.csv. Each displayed row is a measured size. Ratio values above1 mean SME is faster. These results describe this machine, compiler, timing protocol and input reuse pattern. No per-optimization or isolated-instruction-set uplift is claimed.')
    xs,widths=[49,152,384,625],[90,215,219,287]
    for x,w,t in zip(xs,widths,['N','SME GFLOP/s','Accelerate GFLOP/s','SME / Accelerate']): s.text(x+7,153,w-10,t,16,MUTED,bold=True)
    for i,n in enumerate((128,512,1024,2048,4096)):
        y=197+i*44;s.rect(48,y-5,864,41,WHITE if i%2==0 else '#EDF1F6',radius=3)
        vals=[str(n),f'{d.value(n,"sme"):.0f}',f'{d.value(n,"accelerate"):.0f}',f'{d.ratio(n):.2f}×']
        for j,(x,w,val) in enumerate(zip(xs,widths,vals)): s.text(x+7,y+4,w-10,val,22,PINK if j in(1,3) else ORANGE if j==2 else INK,bold=j==1)
    s.band('One final SME backend, one Accelerate reference, and no experimental implementations.')

    s=new('04 / Reproduce','Build, verify, benchmark, and plot',
          'The project keeps the implementation and its measurement path small.',
          'Sources: README.md · build.py · check.py · benchmark.py · plot.py · validation/cleanup/',
          'The build records source and binary hashes and audits generated matrix instructions. Tests verify all packing destinations and padding, compare both backends to an independent scalar double reference, and check repeated overwrites and output guards with ASan/UBSan. Benchmark jobs check results before and after timing. The cleanup validation compares the same optimized algorithm before and after repository simplification using randomized interleaving; its measurements are separate evidence rather than another runnable backend. Scope remains square FP32 multiples of32, one thread and reused inputs. Rebuild slides from existing saved data without running GEMM.')
    s.code(48,145,563,'''
        python3 build.py
        python3 check.py
        python3 benchmark.py --suite full --output results/run1
        python3 plot.py results/run1
    ''',12.5,121)
    s.code(48,288,563,'''
        # Existing results only; no benchmark execution:
        python3 slides/build_slides.py
    ''',12.5,78)
    s.card(634,145,278,264,'Evidence','ASan / UBSan tests\nIndependent reference\nInput and binary hashes\nDisassembly checks\nMeasured cleanup validation',BLUE,18)
    s.band('The final SME design combines efficient arithmetic, packing, and data reuse.')

    s=new('References','Follow the implementation back to primary sources',
          'Architecture/API references and the exact current source files.',
          'Primary references checked 13 November 2026 · performance charts use local measured data',
          '[1] Arm C Language Extensions: '+URLS['ACLE']+'\n[2] University Jena, Hello SME microbenchmarks: '+URLS['Jena']+'\n[3] Apple Accelerate: '+URLS['Apple']+'\n[4] Apple BLAS single-thread control: '+URLS['Threads']+'\nLocal sources: src/sme.c, src/pack.c, src/gemm.h, src/accelerate.c, main.c, benchmark.py. No external microbenchmark measurements are substituted for local results.')
    for i,(title,desc,url) in enumerate([
        ('[1] Arm C Language Extensions','Streaming state, ZA attributes, intrinsics',URLS['ACLE']),
        ('[2] Hello SME / University Jena','FMOPA, SVL and logical tile views',URLS['Jena']),
        ('[3] Apple Accelerate','BLAS and runtime implementation selection',URLS['Apple']),
        ('[4] Apple BLAS threading','Explicit single-thread execution control',URLS['Threads'])]):
        y=145+i*67;s.text(53,y,435,title,18,TEAL,bold=True,link=url);s.text(53,y+27,439,desc,14,MUTED)
    s.card(524,145,388,261,'Local reading order','src/gemm.h → interface and workspace\nsrc/pack.c → final A/B preparation\nsrc/sme.c → final SME kernels\nsrc/accelerate.c → library reference\nexamples/m4-pro/ → current results\nvalidation/cleanup/ → equivalence evidence',BLUE,15.5)
    s.band('Every plotted performance value can be traced to the saved benchmark records.')
    return deck


def export(deck, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Pt(W), Pt(H)
    prs.core_properties.title = "SME and Accelerate on Apple M4 Pro"
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


def preview(out, count):
    """Optional single-image overview; no individual preview files are retained."""
    import pymupdf
    doc=pymupdf.open(out/'sme-on-apple-m4-pro.pdf')
    assert len(doc)==count
    width,height,gap=480,270,18
    rows=math.ceil(count/3)
    sheet=Image.new('RGB',(3*(width+gap)+gap,rows*(height+35+gap)+gap),'#E5E9EF')
    draw=ImageDraw.Draw(sheet)
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.5,.5),alpha=False)
        img=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        x,y=gap+(i%3)*(width+gap),gap+(i//3)*(height+35+gap)
        sheet.paste(img,(x,y));draw.text((x,y+height+8),f'Slide {i+1:02d}',fill=INK)
    sheet.save(out/'overview.png')
    doc.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--results',type=Path,help='Existing full suite; default: <repo>/examples/m4-pro')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--preview',action='store_true',help='Also create overview.png; requires PyMuPDF')
    args=parser.parse_args()
    register_fonts()
    repo,out=args.repo.resolve(),args.output.resolve()
    result=(args.results or repo/'examples/m4-pro').resolve()
    out.mkdir(parents=True,exist_ok=True)
    data=Data(repo,result)
    sys.path.insert(0,str(repo))
    import plot
    with tempfile.TemporaryDirectory(prefix='sme-slide-chart-') as temporary:
        chart=Path(temporary)/'comparison.png'
        fig=plot.figures(result)
        fig.savefig(chart,dpi=180,facecolor='white')
        plt.close(fig)
        deck=make_slides(data,{'comparison':chart})
        export(deck,out)
    manifest={'data_source':str(result.relative_to(repo)) if result.is_relative_to(repo) else str(result),
              'input_sha256':{name:sha(result/name) for name in ['summary.csv','speedups.csv','raw.jsonl','environment.json','status.json']},
              'source_sha256':data.env['build']['source_sha256'],'slides':len(deck),'jobs':data.jobs,'samples':data.samples,
              'checks':['Complete full suite','Throughput formula','Paired medians and IQR against raw jobs','Measured source hashes','Text widths, boundaries and collisions']}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.preview: preview(out,len(deck))
    print(f'Created {len(deck)} slides: PDF, editable PPTX and speaker notes.')
    print(f'Validated {data.jobs} jobs and {data.samples} samples. No benchmark was run.')


if __name__=='__main__':
    main()
