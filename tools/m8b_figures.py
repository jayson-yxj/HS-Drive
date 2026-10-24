"""Static diagnostic figures from frozen outputs, never used for model selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from PIL import Image
from tools.m8b_common import ROOT,provenance,write,asset
from datasets.m8b_petr_adapter import corners_hs,A,ORDER
from diagnostics.m8b_metrics import project_bounds

OUT=Path('reports/m8b_figures')
def save(fig,name):
    p=OUT/(name+'.png');assert not p.exists();fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig)
def main():
    OUT.mkdir(exist_ok=False)
    ff=json.loads((ROOT/'evaluation_ledger.json').read_text())['frames'];idx={x['token']:x for x in json.loads((ROOT/'evaluation_index.json').read_text())};risk=json.loads(Path('reports/m8b_risk_coverage.json').read_text());lat=json.loads(Path('reports/m8b_real_rgb_latency.json').read_text());det=json.loads(Path('reports/m8b_detection_metrics.json').read_text())
    # Display rules: deterministic diagnostic examples, NOT model/threshold selection.
    tokens=[next(f['token'] for f in ff if not f['core40'] and any(g['type']==kind for g in f['gts'])) for kind in ['vehicle','person','cone']]
    tokens+=[risk['high_score_severe_examples'][0]['token']]
    seen=[]
    for token in tokens:
        if token in seen:continue
        seen.append(token);f=next(f for f in ff if f['token']==token);p=json.loads((ROOT/'forward'/(token+'.json')).read_text());packet=json.loads(Path(idx[token]['packet']['path']).read_text())
        fig,axes=plt.subplots(2,6,figsize=(24,8))
        for ci,c in enumerate(packet['cameras']):
            im=np.array(Image.open(c['path']).convert('RGB'));crop=np.array(Image.fromarray(im).resize((1408,844))).copy()[332:844]
            for row,img in [(0,im),(1,crop)]:
                ax=axes[row,ci];ax.imshow(img);ax.set_title(c['name']+(' original' if row==0 else ' official crop'));ax.axis('off')
                if row==0:ax.axhline(332/2.2,color='yellow',lw=1)
                for g in f['gts']:
                    if ci not in [v['camera_index'] for v in g['views']]:continue
                    b=project_bounds(np.array(g['corners']),np.array(c['K']),np.array(c['T_cam_to_ego']),augmented=bool(row))
                    if b is not None:ax.add_patch(Rectangle(b[:2],*(b[2:]-b[:2]),fill=False,color='lime',lw=1.3))
                # Display only top10 >=.1, while all300 retained in raw/evaluation.
                for pi in np.argsort(-np.array(p['scores']))[:10]:
                    if p['scores'][pi]<.1:continue
                    b=project_bounds(corners_hs(p['boxes'][pi]),np.array(c['K']),np.array(c['T_cam_to_ego']),augmented=bool(row))
                    if b is not None:ax.add_patch(Rectangle(b[:2],*(b[2:]-b[:2]),fill=False,color='orangered',lw=.8))
        fig.suptitle(token+' '+f['sample_id']+' | green: recorded visible GT; red: top10 score>=0.1 (display only); yellow: crop boundary\nNo mesh truth / no control authorization',fontsize=12)
        save(fig,token+'_six_camera_crop')
        fig,ax=plt.subplots(figsize=(9,9))
        active=[i for i,s in enumerate(p['scores']) if s>=.1]
        for i in active:
            b=p['boxes'][i];xy=corners_hs(b)[[0,2,6,4],:2];ax.add_patch(Polygon(xy,fill=False,color='orangered',alpha=.3,lw=.7))
        for g in f['gts']:
            xy=np.array(g['corners'])[[0,2,6,4],:2];ax.add_patch(Polygon(xy,fill=False,color='green',lw=2));ax.text(g['box'][0],g['box'][1],g['type'],color='green',fontsize=8)
        for r in f['curves']['0.1']['conditional']:
            b=p['boxes'][r['prediction_index']];g=f['gts'][r['gt_index']]['box'];ax.plot([b[0],g[0]],[b[1],g[1]],'b--',lw=1)
        ax.scatter([0],[0],c='black',marker='^');ax.set(xlim=(-45,45),ylim=(-45,45),xlabel='HS x forward (m)',ylabel='HS y left (m)',title=token+' all score>=0.1 boxes; blue: projection association\nGT green; prediction red; no prediction does not mean Free');ax.set_aspect('equal');ax.grid(alpha=.2);save(fig,token+'_bev')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for s in ['val','calibration','sealed']:
        c=risk['results'][s]['curve'];axes[0,0].plot([x['threshold'] for x in c],[x['natural']['recall'] for x in c],'o-',label=s)
        axes[0,1].plot([x['conditional_coverage'] for x in c],[x['xy']['p95'] or np.nan for x in c],'o-',label=s)
    axes[0,0].set(xlabel='Fixed score threshold',ylabel='Natural center2m recall');axes[0,0].legend()
    axes[0,1].set(xlabel='Projection-associated coverage',ylabel='XY P95 (m)');axes[0,1].axhline(5,c='r',ls='--');axes[0,1].legend()
    for s,col in [('train','gray'),('val','blue'),('calibration','orange'),('sealed','purple')]:
        rr=[r for f in ff if f['split']==s for r in f['curves']['0.0']['conditional']];axes[1,0].scatter([r['score'] for r in rr],[r['xy'] for r in rr],label=s,s=12,alpha=.6)
    axes[1,0].set(xlabel='Raw score (not safety probability)',ylabel='XY error (m)');axes[1,0].legend()
    kk=[k for k in lat['steady_stage_ms'] if k!='total_e2e_ms'];vv=[lat['steady_stage_ms'][k]['p50'] for k in kk];axes[1,1].barh(kk,vv);axes[1,1].set(xlabel='P50 stage ms (synchronized)',title='End-to-end P50 %.1fms'%lat['steady_stage_ms']['total_e2e_ms']['p50'])
    fig.tight_layout();save(fig,'quality_risk_latency')
    fig,ax=plt.subplots(figsize=(8,4));cats=['vehicle','person','cone'];vals=[det['strata']['class/'+c] for c in cats];ax.bar(cats,[v['recall'] for v in vals]);ax.set_ylim(0,1);ax.set_ylabel('Natural center2m recall, score>=0.1')
    for i,v in enumerate(vals):ax.text(i,v['recall']+.03,'TP %d / GT %d'%(v['TP'],v['gt']),ha='center')
    ax.set_title('Historical diagnostic support, not new blind generalization');save(fig,'class_recall')
    write('reports/m8b_visualization_manifest.json',dict(**provenance('OfflineDiagnosticFigures'),selection='Earliest supplemental frame per class plus highest-score severe projection-associated error; post-inference illustrations only, no parameter decisions',tokens=seen,files=[asset(p) for p in sorted(OUT.glob('*.png'))],limits='Display crops PIL interpolation approximate for human view; model used official transforms. Projection rectangles not mask overlap or precise mesh. Top10 display filter only; all300 retained.'))
if __name__=='__main__':main()
