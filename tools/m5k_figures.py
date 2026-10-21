"""Standalone figures from frozen experimental outputs."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5k_common import rows,write,asset
from tools.m5f_common import inputs

def main():
 out=Path('reports/m5k_figures');out.mkdir(exist_ok=True);assets=[]
 fig,axes=plt.subplots(1,2,figsize=(12,4));labels=['original_p8','native16_to_p8','replay101_p8','replay102_p8'];data=json.loads(Path('reports/m5k_alignment_test.json').read_text())
 for ax,field,title in zip(axes,['expected_mae','space1'],['Actor expected MAE (m)','Actor same-layer 1m probability']):
  for j,band in enumerate(['near','10_20','20_35']):
   vals=[next((r[field]for r in data['depth']if r['method']==m and r['scope']=='actor'and r['band']==band),np.nan)for m in labels];ax.bar(np.arange(4)+j*.25,vals,width=.25,label=band)
  ax.set_xticks(np.arange(4)+.25,labels,rotation=15);ax.set_title(title);ax.legend()
 fig.tight_layout();p=out/'m5k_depth_distribution.png';fig.savefig(p,dpi=150);assets.append(asset(p));plt.close(fig)
 samp=json.loads(Path('reports/m5k_sampling_test.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,seed in zip(axes,[101,102]):
  for dep in ['oracle','original','aligned']:
   values=[next(r['recall1']for r in samp['records']if r['seed']==seed and r['depth']==dep and r['sampling']==m)for m in ['p8','full_s4','adaptive','uniform']];ax.plot(['P8','S4','adaptive','uniform'],values,'o-',label=dep)
  ax.set_title(f'All-target 1m recall, Mobile {seed}');ax.set_ylim(0,1);ax.legend()
 fig.tight_layout();p=out/'m5k_sampling.png';fig.savefig(p,dpi=150);assets.append(asset(p));plt.close(fig)
 ix=json.loads(Path('runs/m5k_alignment_test/raw.json').read_text());lookup={r['sample_id']:r['path']for r in ix['index']};rr=rows('test')
 for group in sorted({r['group']for r in rr}):
  row=next(r for r in rr if r['group']==group and r['configuration']==1);z=np.load(lookup[row['sample_id']]);x=inputs(row,'cpu');fig,ax=plt.subplots(2,3,figsize=(13,8));ax[0,0].imshow(x['images'][0,0].permute(1,2,0).numpy());ax[0,0].set_title('Front RGB; fixed config 1');ax[0,1].imshow(z['target'].max(0),origin='lower');ax[0,1].set_title('Reliable visible target surface');ax[0,2].imshow(z['valid'].max(0),origin='lower');ax[0,2].set_title('Reference valid (evaluation only)')
  for aa,key in zip(ax[1],['original_p8_101','native16_to_p8_101','oracle_p8_101']):aa.imshow(z[key].max(0),vmin=0,vmax=1,origin='lower');aa.set_title(key+' (height max display)')
  fig.tight_layout();p=out/(group+'_m5k_spatial.png');fig.savefig(p,dpi=130);assets.append(asset(p));plt.close(fig)
 costs=json.loads(Path('reports/m5k_input_pipeline.json').read_text());fig,ax=plt.subplots(figsize=(10,4))
 for seed in [101,102]:
  r=[r for r in costs['records']if r['seed']==seed];ax.plot([v['mode']for v in r],[v['complete_pipeline_ms']['p95']for v in r],'o-',label=str(seed))
 ax.axhline(50,color='gray',linestyle='--');ax.set_ylabel('Measured complete pipeline P95 ms');ax.legend();fig.tight_layout();p=out/'m5k_input_latency.png';fig.savefig(p,dpi=150);assets.append(asset(p));plt.close(fig)
 write('reports/m5k_visualizations.json',dict(kind='plots_from_actual_frozen_results',assets=assets,limits='Heightmax is display only; quantitative matching sameheight. Representative configuration predefined1, not best-case selection.'))
if __name__=='__main__':main()
