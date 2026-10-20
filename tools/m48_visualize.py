"""Figures from saved actual measurements and CUDA predictions; no synthetic art."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json,file_hash


def read(p):return json.loads(Path(p).read_text())
def main():
    root=Path('reports/m48_figures');root.mkdir(exist_ok=False)
    pilot=read('reports/m48_occupancy_pilot.json')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for r in pilot['records']:
        for ax,key in zip(axs,['surface','free','semantic']):
            ax.plot([x['step'] for x in r['history']],[x['tasks'][key] for x in r['history']],label=r['name']);ax.set(title=key,xlabel='optimizer step',ylabel='masked balanced BCE');ax.set_yscale('log');ax.legend()
    fig.tight_layout();fig.savefig(root/'pilot_losses.png');plt.close(fig)
    with np.load('reports/m48_pilot_artifacts/labels.npz') as z:labels={k:z[k] for k in z.files}
    with np.load('reports/m48_pilot_artifacts/rgb_bev_predictions.npz') as z:pred={k:z[k] for k in z.files}
    for task,target,mask in [('surface','surface_evidence','surface_valid'),('free','free_ray_evidence','free_evidence_valid')]:
        fig,axs=plt.subplots(2,4,figsize=(14,7))
        for layer in range(4):
            for row,values in [(0,labels[target][0,layer]),(1,pred[task][0,layer])]:
                ax=axs[row,layer];ax.imshow(np.where(labels[mask][0,layer],values,np.nan),origin='lower',vmin=0,vmax=1,cmap='viridis');ax.set_title(('target' if row==0 else 'RGB prediction')+f' Z{layer}');ax.set_xlabel('Y index');ax.set_ylabel('X index')
        fig.suptitle(task+' event; gray/blank unknown is NOT a negative');fig.tight_layout();fig.savefig(root/(task+'_prediction.png'));plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for name in ['v1_b0.0_r0','v1_b0.0001_r0','v1_b0.005_r0','hold_at_002','hold_at_008']:
        rows=[json.loads(s) for s in Path('reports/m48_dynamics_artifacts',name+'.jsonl').read_text().splitlines()]
        t=np.array([r['elapsed'] for r in rows]);v=np.array([r['long_speed'] for r in rows]);a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t[:-1])
        for ax,x,y,title in [(axs[0,0],t,v,'Signed speed m/s'),(axs[0,1],t[1:],a,'Raw acceleration m/s²'),(axs[0,2],t[2:],j,'Raw Jerk m/s³'),(axs[1,0],t,[r['command']['brake'] for r in rows],'Command brake'),(axs[1,1],t,[r['hold'] for r in rows],'Actual protocol HOLD')]:
            ax.plot(x,y,label=name);ax.set_title(title);ax.set_xlabel('simulation s')
        axs[1,2].plot(t,v,label=name);axs[1,2].set(xlim=(2.4,3.6),ylim=(-.12,.2),title='Low-speed transition; raw speed')
    axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'dynamics_response.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7))
    for path,label in [('reports/m47_routes_verified_artifacts/m47_junction_turn.jsonl','M47 stalled'),('reports/m48_junction_pilot_v3_artifacts/m48_junction_turn.jsonl','M48 full horizon'),('reports/m48_junction_short_artifacts/m48_junction_turn.jsonl','M48 short diagnostic')]:
        rows=[json.loads(s) for s in Path(path).read_text().splitlines()];ax.plot([r['x'] for r in rows],[r['y'] for r in rows],label=label)
    route=read('reports/m45_oracle_driving_artifacts/junction_turn_route.json')['route_world_hs'];route=np.array(route);ax.plot(route[:,0],route[:,1],'k--',alpha=.4,label='route');ax.axis('equal');ax.legend();ax.set(xlabel='HS world X m',ylabel='HS world Y m',title='Actual CARLA junction tracks');fig.tight_layout();fig.savefig(root/'junction_tracks.png');plt.close(fig)
    # Show corrected coexistence for an originally audited cell with measured geometry.
    audit=read('reports/m47_false_free_audit.json')['records'];r=audit[0];lo,hi=np.array(r['bounds_xyz_m']);fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d')
    for ray in r['free_rays']:ax.plot(*np.array([ray['entry'],ray['exit']]).T,c='green')
    pts=np.array([x['endpoint'] for x in r['heldout_surface']]);ax.scatter(*pts.T,c='red',label='surface observation')
    for z in np.arange(lo[2],hi[2]+.001,.25):ax.plot([lo[0],hi[0]],[lo[1],lo[1]],[z,z],c='gray',alpha=.5)
    ax.set(xlabel='X m',ylabel='Y m',zlabel='Z m',title=r['id']+'\nFree-ray and surface evidence may coexist');ax.legend();fig.tight_layout();fig.savefig(root/'mixed_cell_3d.png');plt.close(fig)
    atomic_json(root/'manifest.json',dict(source_sha256=file_hash(__file__),figures={p.name:file_hash(p) for p in root.glob('*.png')}))
if __name__=='__main__':main()
