"""Scientific artifacts from recorded replay, not generated illustrative data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m7d_common import read


def main():
    out=Path('reports/m7d_figures');out.mkdir(exist_ok=False)
    ds=[read(p) for p in sorted(Path('reports/m7d_replays').glob('*_stream.json'))]
    labels={None:0,'D_stop':1,'E_slow_follow':2,'A_keep_route':3,'B_left':4,'C_right':5}
    fig,axs=plt.subplots(4,2,figsize=(15,11),sharex=True,sharey=True)
    for ax,d in zip(axs.flat,ds):
        rs=d['records'];x=[r['input_index']*.05 for r in rs];ax.step(x,[labels[r['selected']] for r in rs],where='post');ax.set_title(d['episode']);ax.set_yticks(list(labels.values()),['Reject','Stop','Slow','Keep','Left','Right']);ax.grid(alpha=.2)
    fig.supxlabel('Original episode time (s)');fig.suptitle('Offline decisions at measured states (no trajectory execution)');fig.tight_layout();fig.savefig(out/'m7d_decision_timeline.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,8))
    for ep in ['dev_constant_r3','dev_braking_r20']:
        for mode,style in [('baseline','--'),('cold',':'),('stream','-')]:
            p=Path(f'reports/m7d_replays/{ep}_{mode}_clean.json')
            if not p.exists():continue
            d=read(p);rs=[r for r in d['records'] if 'timing_ms'in r];axs[0].plot([r['input_index']*.05 for r in rs],[r['timing_ms']['total'] for r in rs],style,label=ep+' '+mode)
    axs[0].axhline(200,color='red',label='200ms budget');axs[0].set_ylabel('Compute wall time (ms)');axs[0].legend(fontsize=8)
    for ep in ['dev_braking_r20','final_braking_r30']:
        rows=[json.loads(x) for x in Path(f'reports/m7d_episodes/{ep}/trace.jsonl').read_text().splitlines()];t=[r['sim_time']-rows[0]['sim_time'] for r in rows]
        axs[1].plot(t,[np.linalg.norm(r['actors'][0]['velocity']) for r in rows],label=ep+' measured lead')
        axs[1].plot(t,[r['commands']['lead_target'] for r in rows],'--',label=ep+' scripted target')
    axs[1].set_ylabel('Actual vs command speed (m/s)');axs[1].set_xlabel('Original simulation time (s)');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'m7d_latency_and_actual_motion.png',dpi=160);plt.close(fig)
    d=read('reports/m7d_replays/dev_constant_r3_stream.json');slow=next(r for r in d['records'] if r['selected']=='E_slow_follow');idx=slow['input_index'];future=read(f'reports/m7d_evaluator/dev_constant_r3/{idx:05d}.json');fig,ax=plt.subplots(figsize=(9,7));origin=np.array(slow['request']['scene']['ego_xy_yaw_speed'][:2]);yaw=slow['request']['scene']['ego_xy_yaw_speed'][2];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    for e in slow['evaluations']:
        pts=(np.array(e['candidate']['points'])[:,:2]-origin)@rot;ax.plot(pts[:,0],pts[:,1],label=e['name']+(' accepted' if e['accepted'] else ' rejected'))
    pts=(np.array([r['actors'][0]['xy'] for r in future['samples']])-origin)@rot;ax.plot(pts[:,0],pts[:,1],'k--',label='Measured lead future (evaluator only)');ax.set_aspect('equal');ax.set_xlabel('Ego-origin forward (m)');ax.set_ylabel('Left (m)');ax.legend(fontsize=8);ax.set_title('Actual-state slow candidate / finite recorded future');fig.tight_layout();fig.savefig(out/'m7d_slow_spatial_reference.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
