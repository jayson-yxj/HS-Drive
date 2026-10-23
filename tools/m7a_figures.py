"""Static research figures from frozen stage artifacts."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator.oracle_driver import footprint
from tools.m7a_common import read


def main():
    out=Path('reports/m7a_figures');out.mkdir(exist_ok=False)
    rows=read('reports/m7a_planner_validation.json')['records']
    chosen=[next(r for r in rows if r['road']==67 and r['kind']==k) for k in ['static_bypass','blocked_near_reject','crossing_conflict','crossing_clear_time']]
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,r in zip(axes.flat,chosen):
        route=np.array(r['request']['route']);origin=route[0,:2]
        ax.plot(*(route[:,:2]-origin).T,'k:',label='route')
        for c,e in zip(r['candidates'],r['evaluations']):
            p=np.array(c['points'])
            ax.plot(*(p[:,:2]-origin).T,label=c['name']+(' accept' if e['planning_feasible'] else ' reject'))
            if c['name']=='B_left':ax.plot(*(np.array(c['guide_path'])[:,:2]-origin).T,alpha=.3)
        for o in r['request']['scene']['boxes']:
            poly=footprint(*o['xy'],o['yaw'],o['half_length'],o['half_width'])-origin
            ax.fill(*poly.T,alpha=.4,color='red');v=np.array(o['velocity'])
            ax.arrow(*(np.array(o['xy'])-origin),*(v*2),color='red',width=.03)
        ax.set_title(r['kind']+' / '+str(r['decision']['selected']));ax.axis('equal');ax.grid();ax.legend(fontsize=7)
        ax.set_xlabel('HS world delta x m');ax.set_ylabel('HS world delta y m')
    fig.suptitle('M7-A offline Town03 + synthetic actors; no vehicle execution');fig.tight_layout()
    fig.savefig(out/'m7a_candidates.png',dpi=150);plt.close(fig)
    r=read('reports/m7a_shadow_rgb_forward_audited.json')['records'][0]
    p=read(next(x['path'] for x in read('reports/m7a_shadow_inference_manifest.json')['packets'] if read(x['path'])['sample_id']==r['sample_id']))
    canvas=Image.new('RGB',(1280,3*414),'white')
    for i,c in enumerate(p['cameras']):
        im=Image.open(c['path']).convert('RGB');draw=ImageDraw.Draw(im)
        for h in r['hypotheses']:
            if h['camera']==c['name']:
                x,y,w,hh=h['region'];draw.rectangle((x,y,x+w,y+hh),outline='yellow',width=2)
        xx=(i%2)*640;yy=(i//2)*414;canvas.paste(im,(xx,yy));ImageDraw.Draw(canvas).text((xx+8,yy+386),c['name']+' | frame '+str(c['frame_id'])+' | CandidateNoGo',fill='black')
    canvas.save(out/'m7a_six_camera_shadow.png')
    fig,ax=plt.subplots(figsize=(10,4))
    for split in ['development_v2','validation','final']:
        d=read('reports/m7a_planner_'+split+'.json');v=[r['total_ms'] for r in d['records']]
        ax.plot(sorted(v),label=split)
    ax.axhline(200,color='red',ls='--',label='5Hz period reference')
    ax.set_ylabel('offline complete candidate audit ms');ax.set_xlabel('sorted scenario, not time series');ax.grid();ax.legend()
    ax.set_title('Audit cost is not production control latency')
    fig.tight_layout();fig.savefig(out/'m7a_planning_latency.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
