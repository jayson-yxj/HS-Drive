"""Actual frozen source-ray traces and paired BEV visualization."""
import json
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5j_common import rows,write
from tools.m5f_common import inputs,input_path,pixel_label
from datasets.labels import depth_bin_edges
from simulator.recorder import file_hash

def main():
    dest=Path('reports/m5j_figures');dest.mkdir(exist_ok=False);audit=json.loads(Path('reports/m5j_evidence_attribution_test.json').read_text());rr={r['sample_id']:r for r in rows('test')};chosen=[]
    for tag in ['A_2D_wrong','B_target_depth_misprojected','D','E']:
        candidates=[r for r in audit['examples']if r['seed']==101 and (r['domain']=='unknown'if tag=='E'else r['domain']=='background')]
        r=max(candidates,key=lambda r:r['low_q_mass']if tag=='D'else r['mass']if tag=='E'else r['source_mass'].get(tag,0));chosen.append((tag,r))
    edges=depth_bin_edges();centers=(edges[:-1]+edges[1:])/2
    for tag,r in chosen:
        row=rr[r['sample_id']];ray=r['top_rays'][0];ci=row['camera_names'].index(ray['camera']);inp=json.loads(input_path(row).read_text());rgb=cv2.cvtColor(cv2.imread(inp['images'][ci]['path']),cv2.COLOR_BGR2RGB);d=pixel_label(row);idx=json.loads(Path('runs/m5j_test/index.json').read_text())['records'];cache=np.load(next(x['cache']for x in idx if x['sample_id']==r['sample_id']));fig,ax=plt.subplots(2,3,figsize=(15,8));ax[0,0].imshow(rgb);u,v=ray['pixel_uv'];ax[0,0].scatter([u],[v],c='red',s=25);ax[0,0].set_title(r['group']+' '+ray['camera']);ax[0,1].imshow(cache['101_pixel'][ci],vmin=0,vmax=1,cmap='magma');ax[0,1].set_title('RGB pixel probability');ax[0,2].imshow(np.where(d['valid'][ci],d['target'][ci].astype(float),np.nan));ax[0,2].set_title('Reliable visible target / Unknownwhite');ax[1,0].plot(centers,ray['probabilities']);ax[1,0].axvline(ray['gt_depth'],color='red');ax[1,0].set_title('Top source ray depth distribution / GTred');z,x,y=r['cell'];ax[1,1].imshow(cache['101_raw'][z],origin='lower',vmin=0,vmax=1,cmap='magma');ax[1,1].scatter([y],[x],c='cyan',s=25);ax[1,1].set_title('Raw evidence layer '+str(z));ax[1,2].bar(list(r['source_mass']),list(r['source_mass'].values()));ax[1,2].tick_params(axis='x',rotation=35);ax[1,2].set_title('Additive source mass; C/D mayoverlap');fig.suptitle(tag+' diagnostic source attribution, not physicalFP certification');fig.tight_layout();fig.savefig(dest/('m5j_trace_'+tag+'.png'),dpi=140);plt.close(fig)
    report=json.loads(Path('reports/m5j_aggregation_test.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,5))
    for r in report['curves']:
        c=r['curve'];ax[0].plot([v['recall']for v in c],[v['precision']for v in c],label=f"{r['seed']}_{r['method']}");ax[1].plot([v['fpr']for v in c],[v['recall']for v in c])
    ax[0].set(xlabel='Exact surface Recall',ylabel='Precision',title='Testcurves descriptive, thresholdslockedVal');ax[0].legend();ax[1].set(xscale='log',xlabel='Confirmedbackground FPR',ylabel='Recall');fig.tight_layout();fig.savefig(dest/'m5j_pr.png',dpi=150);plt.close(fig)
    write('reports/m5j_visualizations.json',dict(kind='offline_actual_frozen_trace_figures',files={str(p):file_hash(p)for p in dest.iterdir()},examples=chosen,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
