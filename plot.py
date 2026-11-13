#!/usr/bin/env python3
"""Plot saved SME/Accelerate results; never execute GEMM."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'build/plot-cache'))
LABELS={'sme':'SME','accelerate':'Accelerate'}
COLORS={'sme':'#db2777','accelerate':'#c76b13'}


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def figures(directory):
    """Return throughput and paired speedup panels for reuse in the slide deck."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    summary,speedups=rows(directory/'summary.csv'),rows(directory/'speedups.csv')
    if not summary:
        raise ValueError('No timings: verification-only runs cannot be plotted')
    if not {r['backend'] for r in summary} <= LABELS.keys():
        raise ValueError('Expected only SME and Accelerate results')
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    throughput,speedup=axes
    for backend in LABELS:
        rates=sorted((r for r in summary if r['backend']==backend),key=lambda r:int(r['size']))
        if rates:
            throughput.plot([int(r['size']) for r in rates],[float(r['gflops'])/1000 for r in rates],
                            'o-',color=COLORS[backend],label=LABELS[backend],markersize=4)
    ratios=sorted((r for r in speedups if r['baseline']=='accelerate' and r['backend']=='sme'),key=lambda r:int(r['size']))
    if ratios:
        x=[int(r['size']) for r in ratios]
        speedup.plot(x,[float(r['speedup_median']) for r in ratios],'o-',color=COLORS['sme'],label='SME',markersize=4)
        speedup.fill_between(x,[float(r['speedup_p25']) for r in ratios],[float(r['speedup_p75']) for r in ratios],color=COLORS['sme'],alpha=.13)
    speedup.axhline(1,ls='--',color=COLORS['accelerate'],label='Accelerate = 1×')
    throughput.set_title('Throughput including packing')
    throughput.set_ylabel('Throughput (TFLOP/s)')
    speedup.set_title('Speedup = Accelerate time / SME time')
    speedup.set_ylabel('Speedup vs Accelerate (×)')
    sizes=sorted({int(r['size']) for r in summary})
    for ax in axes:
        ax.set_xscale('log',base=2)
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Square matrix dimension N')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=9)
    cpu=json.loads((directory/'environment.json').read_text()).get('cpu','Apple Silicon')
    fig.suptitle(f'{cpu} · FP32 GEMM · one thread',fontsize=15,fontweight='bold')
    fig.text(.04,.025,'Packing included. Shading: IQR of paired-round speedups, not a confidence interval.',fontsize=9)
    fig.tight_layout(rect=(.01,.06,1,.94))
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    directory=args.results.expanduser().resolve()
    fig=figures(directory)
    output=directory/'plots';output.mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):
        target=output/f'comparison.{suffix}'
        fig.savefig(target,dpi=180,facecolor='white')
        if suffix=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
        print(target)


if __name__=='__main__':
    main()
