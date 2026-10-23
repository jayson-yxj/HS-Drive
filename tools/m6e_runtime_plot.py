"""All trial interval distributions grouped by actual Physics, no tail exclusion."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m6e_common import write,provenance,asset

def main():
    fig,ax=plt.subplots(figsize=(13,6));events=[]
    for i,p in enumerate(sorted(Path('reports/m6e_trials').glob('*/trace.jsonl'))):
        r=[json.loads(s) for s in p.read_text().splitlines()];interval=np.diff([q['timeline']['apply_control'] for q in r])*1000;k=int(interval.argmax())+1;a=r[k-1]['timeline'];b=r[k]['timeline'];version=r[0]['physics_version']
        ax.scatter([i]*4,np.quantile(interval,[.5,.95,.99,1]),color='tab:blue' if version=='P0' else 'tab:orange',s=[8,15,20,30])
        events.append(dict(name=p.parent.name,physics=version,max_ms=float(interval.max()),frame=r[k]['frame'],trace=asset(p),
            parts_ms=dict(previous_apply_to_tick=(a['tick_start']-a['apply_control'])*1000,previous_tick=(a['tick_end']-a['tick_start'])*1000,tail=(b['previous_sleep_start']-a['tick_end'])*1000,sleep_wakeup=(b['wake_up']-b['previous_sleep_start'])*1000,current_preapply=(b['apply_control']-b['wake_up'])*1000)))
    ax.axhline(50,color='red',ls='--');ax.set_xlabel('Every trial in fixed archive order');ax.set_ylabel('P50/P95/P99/Max apply interval ms');ax.set_title('Blue=P0, orange=P1; all tails retained');ax.grid(alpha=.3);fig.tight_layout();fig.savefig('reports/m6e_figures/m6e_runtime.png',dpi=140);plt.close(fig)
    write('reports/m6e_runtime_tail_events.json',dict(**provenance('offline_actual_fixed_slot_timeline'),trials=events,worst=max(events,key=lambda q:q['max_ms']),
        limits='Clock attribution only. Physics variants and scene order are not randomized OS load conditions; no causal Tick claim.'))
if __name__=='__main__':main()
