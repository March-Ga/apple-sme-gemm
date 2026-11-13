#!/usr/bin/env python3
"""Plot an existing intrinsics benchmark run; never execute GEMM."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"build/matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/"build/plot-cache"))
LABELS = {"sme32": "Original SME32", "sme_opt": "Optimized SME", "accelerate": "Accelerate"}
COLORS = {"sme32": "#007f86", "sme_opt": "#db2777", "accelerate": "#c76b13"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    directory = args.results.expanduser().resolve()
    summary, speedups = rows(directory/"summary.csv"), rows(directory/"speedups.csv")
    if not summary:
        parser.error("No timing results: correctness-only runs cannot be plotted")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        parser.exit(1, "Requires matplotlib: see README.md\n")
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for column, mode in enumerate(("compute", "total")):
        top, bottom = axes[:, column]
        available = {r["backend"] for r in summary if r["mode"] == mode}
        baseline = "accelerate" if "accelerate" in available else "sme32"
        top.set_title("Prepared inputs (packing excluded)" if mode == "compute" else "Packing + computation")
        for backend in LABELS:
            rates = sorted((r for r in summary if r["mode"] == mode and r["backend"] == backend),
                           key=lambda r: int(r["size"]))
            ratios = sorted((r for r in speedups if r["mode"] == mode and r["backend"] == backend
                             and r["baseline"] == baseline), key=lambda r: int(r["size"]))
            if ratios:
                x = [int(r["size"]) for r in ratios]
                top.plot(x, [float(r["speedup_median"]) for r in ratios], "o-",
                         color=COLORS[backend], label=LABELS[backend], markersize=4)
                top.fill_between(x, [float(r["speedup_p25"]) for r in ratios],
                                 [float(r["speedup_p75"]) for r in ratios], color=COLORS[backend], alpha=.13)
            if rates:
                x = [int(r["size"]) for r in rates]
                bottom.plot(x, [float(r["gflops"]) for r in rates], "o-",
                            color=COLORS[backend], label=LABELS[backend], markersize=4)
        top.axhline(1, linestyle="--", color="#334155", linewidth=1, label=f"{LABELS[baseline]} (1×)")
        top.set_ylabel(f"Speedup vs {LABELS[baseline]} (×)")
        bottom.set_ylabel("Throughput (GFLOP/s)")
        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Square matrix dimension N")
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=9)
    env = json.loads((directory/"environment.json").read_text())
    fig.suptitle("GEMM in C with intrinsics · FP32 · single thread", fontsize=16, fontweight="bold", y=.985)
    fig.text(.05, .935, "Optimized SME uses dedicated packing and blocking. Shading: IQR of paired speedups, not a confidence interval.", fontsize=9)
    cpu = env.get("cpu", "Apple Silicon")
    if cpu == "unavailable":
        cpu = env.get("post_run_metadata", {}).get("cpu", "Apple Silicon")
    fig.text(.05, .02, f"{cpu} · Accelerate is a library reference; end-to-end comparison only.", fontsize=9)
    fig.tight_layout(rect=(.02, .045, 1, .91))
    output = directory/"plots"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        target = output/f"comparison.{suffix}"
        fig.savefig(target, dpi=170, facecolor="white")
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
