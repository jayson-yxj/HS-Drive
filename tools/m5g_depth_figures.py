"""Raw depth probability and near/far tradeoff visualizations, no smoothing."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5g_depth_visualizations.json')
    if dest.exists():raise FileExistsError(dest)
    out=Path('reports/m5g_depth_figures');out.mkdir(exist_ok=False);study=json.loads(Path('reports/m5g_depth_training.json').read_text());fig,ax=plt.subplots(1,3,figsize=(15,4))
    for r in study['records']:
        h=[v for v in r['history']if 'validation'in v];label=f"{r['name']}/{r['seed']}"
        for i,key in enumerate(['near_mae','target_spatial_macro','ce_loss']):ax[i].plot([v['step']for v in h],[v['validation'][key]for v in h],marker='o',label=label);ax[i].set(xlabel='Step',title=key)
    ax[0].axhline(study['original_validation']['near_mae']*1.05,color='r',linestyle='--',label='Near gate');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5g_depth_tradeoff.png',dpi=160);plt.close(fig)
    d=json.loads(Path('reports/m5g_depth_test_final.json').read_text());fig,ax=plt.subplots(2,3,figsize=(15,8));bands=['1.5_10','10_20','20_35','35_60','60_80']
    for name,seed in sorted({(r['name'],r['seed'])for r in d['aggregate']}):
        for row,scope in enumerate(['all','actor']):
            rr=[r for r in d['aggregate']if r['name']==name and r['seed']==seed and r['scope']==scope]
            for col,key in enumerate(['expected_mae','same_layer_1m_mass','distribution_expected_space_error']):
                ys=[next(r for r in rr if r['distance']==b)['metrics'][key]['mean']for b in bands];ax[row,col].plot(bands,[np.nan if y is None else y for y in ys],marker='o',label=f'{name}/{seed}');ax[row,col].set_title(scope+' '+key);ax[row,col].tick_params(axis='x',rotation=35)
    ax[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5g_depth_distance_test.png',dpi=160);plt.close(fig)
    shadow=json.loads(Path('reports/m5g_shadow_comparison.json').read_text())['records'];groups=sorted({r['group']for r in shadow})
    for group in groups:
        rr=[r for r in shadow if r['group']==group and r['seed']==101 and r['configuration']==2];fig,ax=plt.subplots(4,6,figsize=(18,10))
        for k,r in enumerate(rr):
            z=np.load(r['path'])
            for layer in range(4):
                ax[layer,k].imshow(z['score'][layer],origin='lower',extent=(-30,30,-20,60),vmin=0,vmax=1,cmap='magma');ax[layer,k].set_title(r['mode']+f' z{layer}',fontsize=7)
                if k==0:ax[layer,5].imshow(z['target'][layer]&z['valid'][layer],origin='lower',extent=(-30,30,-20,60));ax[layer,5].set_title('Rendered reference',fontsize=7)
        fig.suptitle(group+' config2 seed101 (new depth diagnostic fails near gate)');fig.tight_layout();fig.savefig(out/f'm5g_joint_{group}.png',dpi=140);plt.close(fig)
    atomic_json(dest,dict(kind='offline_figures_actual_GPU_probability_results',source_sha256=file_hash(__file__),files={str(p):file_hash(p)for p in out.glob('*.png')},limits='NewDepthfailednearqualitygate; showallseeds/bands, no smoothedmetrics. Fixedconfig2 seed101 notbesttest.'))

if __name__=='__main__':main()
