"""Paired successes, regressions and strict identities, without changing grades."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from datasets.m8b_petr_adapter import corners_hs
from diagnostics.m8b_metrics import project_bounds
from tools.m8d_common import *

def main():
    out=Path('reports/m8d_figures')
    quality=json.loads(Path('reports/m8d_paired_detection_quality.json').read_text())
    identity=json.loads(Path('reports/m8d_paired_identity_support.json').read_text())
    ledger={f['token']:f for f in json.loads((BASELINE/'evaluation_ledger.json').read_text())['frames']}
    evaluation={f['token']:f for f in json.loads((ROOT/'evaluation.json').read_text())['frames']}
    files=[]
    def save(fig,name):
        p=out/name
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);files.append(asset(p))
    groups=quality['all38_groups']
    fig,axes=plt.subplots(2,1,figsize=(14,7))
    x=np.arange(38)
    axes[0].bar(x-.2,[g['A_TP'] for g in groups],.4,label='A original')
    axes[0].bar(x+.2,[g['B_TP'] for g in groups],.4,label='B converter candidate')
    axes[0].set_ylabel('Joint detection TP');axes[0].legend()
    axes[0].set_xticks(x);axes[0].set_xticklabels([g['group'] for g in groups],rotation=90,fontsize=6)
    keys=['class/vehicle','class/person','class/cone','range/0-10','range/10-20','range/20-40']
    y=np.arange(len(keys))
    axes[1].bar(y-.2,[quality['strata'][k]['A_recall'] or 0 for k in keys],.4,label='A')
    axes[1].bar(y+.2,[quality['strata'][k]['B_recall'] or 0 for k in keys],.4,label='B')
    axes[1].set_xticks(y);axes[1].set_xticklabels([k+' n='+str(quality['strata'][k]['GT']) for k in keys])
    axes[1].set_ylim(0,1);axes[1].set_ylabel('Recall / fixed GT');axes[1].legend()
    fig.tight_layout();save(fig,'group_and_strata.png')
    selection={}
    for category,test in [('gained',lambda r:not r['A'] and r['B']),('lost',lambda r:r['A'] and not r['B'])]:
        rows=[r for r in quality['targets'] if test(r)]
        selection[category]=[dict(token=r['token'],gt_index=r['gt_index']) for r in rows]
        for page in range(0,len(rows),6):
            rr=rows[page:page+6]
            fig,axes=plt.subplots(len(rr),2,figsize=(12,3.5*len(rr)),squeeze=False)
            for row,r in enumerate(rr):
                token=r['token'];f=ledger[token];g=f['gts'][r['gt_index']]
                packet=json.loads((BASELINE/'rgb_bundle'/token/'packet.json').read_text())
                view=max(g['views'],key=lambda v:v['retained_pixels']);ci=view['camera_index'];cam=packet['cameras'][ci]
                for col,v in enumerate(['A','B']):
                    pred=json.loads(((BASELINE/'forward'/(token+'.json')) if v=='A' else ROOT/'full'/(token+'_B.json')).read_text())
                    ax=axes[row,col];ax.imshow(Image.open(cam['path']))
                    bb=project_bounds(np.array(g['corners']),np.array(cam['K']),np.array(cam['T_cam_to_ego']),False)
                    if bb is not None:ax.add_patch(plt.Rectangle(bb[:2],*(bb[2:]-bb[:2]),fill=False,color='lime',lw=2))
                    variant=evaluation[token]['variants'][v]
                    pi=next((i for i,j in variant['natural']['pairs'] if j==r['gt_index']),None)
                    status='joint TP'
                    if pi is None:
                        pi=next((i for i,j in variant['projection_pairs'] if j==r['gt_index']),None)
                        status='projection proxy only'
                    if pi is not None:
                        box=project_bounds(corners_hs(pred['boxes'][pi]),np.array(cam['K']),np.array(cam['T_cam_to_ego']),False)
                        if box is not None:ax.add_patch(plt.Rectangle(box[:2],*(box[2:]-box[:2]),fill=False,color='red',lw=1.5))
                    else:status='no association'
                    ax.axhline(332/2.2,color='yellow',ls='--',lw=.7)
                    ax.set_title('%s %s %s GT%d %s %s'%(token,v,category,r['gt_index'],cam['name'],status),fontsize=9)
                    ax.axis('off')
            save(fig,'%s_%02d.png'%(category,page//6))
    common=identity['common_supported']
    fig,axes=plt.subplots(max(1,len(common)),2,figsize=(12,3*max(1,len(common))),squeeze=False)
    for i,key in enumerate(common):
        token,gi=key;packet=json.loads((BASELINE/'rgb_bundle'/token/'packet.json').read_text())
        g=ledger[token]['gts'][gi];ci=max(g['views'],key=lambda v:v['retained_pixels'])['camera_index'];cam=packet['cameras'][ci]
        for col,v in enumerate(['A','B']):
            r=next(r for r in identity['all_rows'][v] if r['token']==token and r['gt_index']==gi)
            pred=json.loads(((BASELINE/'forward'/(token+'.json')) if v=='A' else ROOT/'full'/(token+'_B.json')).read_text())
            ax=axes[i,col];ax.imshow(Image.open(cam['path']))
            for b,color in [(g['box'],'lime'),(pred['boxes'][r['prediction_index']],'red')]:
                bounds=project_bounds(corners_hs(b),np.array(cam['K']),np.array(cam['T_cam_to_ego']),False)
                if bounds is not None:ax.add_patch(plt.Rectangle(bounds[:2],*(bounds[2:]-bounds[:2]),fill=False,color=color))
            ax.set_title('%s %s common identity; XY %.2fm'%(token,v,r['metric_error']['XY_m']))
            ax.axis('off')
    if not common:
        for ax in axes.ravel():ax.text(.2,.5,'No common identity-supported targets');ax.axis('off')
    save(fig,'common_identity.png')
    diff=json.loads(Path('reports/m8d_gpu_output_difference.json').read_text())
    first=diff['rows'][0]
    fig,ax=plt.subplots(figsize=(8,4))
    for key in ['geometry_encoding','sine_encoding','adapted_sine_encoding','final_position_embedding']:
        ax.plot(CFG['A'],[x['absolute']['max'] for x in first['position_by_physical_camera'][key]],marker='o',label=key)
    ax.set_ylabel('Max absolute difference, aligned physical camera')
    ax.set_title('Fixed first sample; position encoding only, not quality');ax.legend(fontsize=7);ax.tick_params(axis='x',rotation=25)
    save(fig,'physical_camera_encoding.png')
    write('reports/m8d_visualization_manifest.json',dict(**provenance('OfflineDiagnosticFigures'),files=files,
        selection=selection,rule='All gained and lost natural GT targets, all common strict identities. No selection by B error.',
        colors='Green recorded GT OBB projection; red associated predicted OBB projection. Does not certify physical identity. Yellow crop boundary.'))
if __name__=='__main__':main()
