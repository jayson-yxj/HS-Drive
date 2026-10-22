"""Monotonic control interval plots retain entire runtime tails."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    paths=sorted(Path('reports/m6c_trials').glob('runtime_*/trace.jsonl'));fig,axes=plt.subplots(2,1,figsize=(12,8))
    for p in paths:
        r=[json.loads(s) for s in p.read_text().splitlines()];v=np.diff([q['timeline']['apply_control'] for q in r])*1000
        axes[0].plot(np.arange(len(v)),v,lw=.6,label=p.parent.name);axes[1].plot(np.sort(v),np.linspace(0,1,len(v)),label=p.parent.name)
    axes[0].axhline(50,color='r',ls='--');axes[0].set_ylabel('Apply interval ms');axes[0].set_xlabel('cycle');axes[0].legend()
    axes[1].set_xlabel('Apply interval ms');axes[1].set_ylabel('Empirical CDF (correlated cycles)');axes[1].axvline(50,color='r',ls='--');axes[1].legend();fig.tight_layout()
    Path('reports/m6c_figures').mkdir(exist_ok=True);fig.savefig('reports/m6c_figures/m6c_runtime_intervals.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
