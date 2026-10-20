"""Aggregate descriptive evidence and generate standalone research figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostics.distributions import BANDS

NAMES=['m4_expanded_joint','m4_far_balanced','m4_road_point']
SHORT=['Joint','Far balanced','Road point']
COLORS=['#2978b5','#e58c27','#22956a']


def means(rows):
    keys=rows[0]
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys if not isinstance(rows[0][k],(dict,list,str))}


def aggregate(r):
    variants=list(r['bev_frames'][0]['variants']);scenes=sorted({f['scene'] for f in r['frames']});result={}
    for sid in ['all',*scenes]:
        rows=[v for v in r['bev_frames'] if sid=='all' or v['scene']==sid]
        rr={}
        for name in variants:
            records=[x['variants'][name] for x in rows]
            cm=np.sum([v['road']['confusion_tn_fp_fn_tp'] for v in records],axis=0);tn,fp,fn,tp=cm.ravel()
            rr[name]={'mass_frame_mean':means([v['mass'] for v in records]),'feature_frame_mean':means([v['features'] for v in records]),
                'surface_frame_mean':means([v['surface_head'] for v in records]),'road_pooled_iou':float(tp/(tp+fp+fn)),
                'mass_layer_totals':np.sum([v['mass']['layer_mass'] for v in records],axis=0).tolist(),
                'oracle_layer_totals':np.sum([v['mass']['layer_oracle_count'] for v in records],axis=0).tolist()}
        ratio=[x['out_of_bev_gt_rays_probability_leaking_in']/max(x['variants']['learned_distribution']['mass']['predicted_mass'],1e-9) for x in rows]
        result[sid]={'frames':len(rows),'variants':rr,'native_surface_frame_mean':means([x['native_surface_head'] for x in rows]),
                     'out_of_bev_leak_fraction_of_predicted_mass_frame_mean':float(np.mean(ratio))}
    correlations={}
    for sid in ['all',*scenes]:
        ids=[i for i,f in enumerate(r['frames']) if sid=='all' or f['scene']==sid]
        x=np.array([r['frames'][i]['depth']['expected_mae_m'] for i in ids]);y=np.array([r['bev_frames'][i]['variants']['learned_distribution']['mass']['normalized_spatial_tv'] for i in ids])
        correlations[sid]={'frames':len(ids),'pearson_depth_mae_vs_mass_tv':float(np.corrcoef(x,y)[0,1]) if x.std()>1e-8 and y.std()>1e-8 else None,
                          'caveat':'descriptive correlated frames; not independent-trial significance or causal attribution'}
    result['correlations']=correlations
    return result


def plots(report,root):
    root.mkdir(parents=True,exist_ok=False);results=report['results'];bands=list(BANDS)[1:];x=np.arange(4)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    metrics=[('expected_mae_m','Expected MAE (m)'),('mode_mae_m','Mode MAE (m)'),('median_mae_m','Median MAE (m)'),('expected_signed_bias_m','Signed bias: prediction - GT (m)'),('normalized_entropy','Normalized categorical entropy'),('gt_bin_probability','GT bin probability')]
    for ax,(key,title) in zip(axes.flat,metrics):
        for name,label,color in zip(NAMES,SHORT,COLORS):
            d=results[name]['distribution']['all_valid']['all'];ax.plot(x,[d[b][key] for b in bands],'-o',color=color,label=label)
        ax.set(xticks=x,xticklabels=bands,title=title);ax.grid(alpha=.2)
        if 'bias' in key:ax.axhline(0,color='gray',lw=.8)
    axes[0,0].legend();fig.suptitle('Fixed validation set, original RGB; identical sampled rays and masks');fig.tight_layout();fig.savefig(root/'depth_decoders_and_uncertainty.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,title in zip(axes,['mass_within_1m','mass_within_3m','mass_within_5m'],['Mass within GT +/-1m','Mass within GT +/-3m','Mass within GT +/-5m']):
        for name,label,color in zip(NAMES,SHORT,COLORS):
            d=results[name]['distribution']['all_valid']['all'];ax.plot(x,[d[b][key] for b in bands],'-o',label=label,color=color)
        ax.set(xticks=x,xticklabels=bands,ylim=(0,1),title=title);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Discrete bin-center probability mass; no within-bin interpolation assumption');fig.tight_layout();fig.savefig(root/'gt_neighborhood_mass.png');plt.close(fig)
    camera_keys=[k for k in results[NAMES[0]]['distribution']['all_valid'] if k.startswith('camera/')]
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    for ax,band in zip(axes,['all','60-80m']):
        for i,(name,label,color) in enumerate(zip(NAMES,SHORT,COLORS)):
            values=[results[name]['distribution']['all_valid'][k][band]['expected_mae_m'] for k in camera_keys]
            ax.bar(np.arange(6)+(i-1)*.24,values,.24,label=label,color=color)
        ax.set(xticks=np.arange(6),xticklabels=[k.split('/')[1] for k in camera_keys],ylabel='Expected MAE (m)',title=band);ax.tick_params(axis='x',rotation=25)
    axes[0].legend();fig.tight_layout();fig.savefig(root/'camera_depth_errors.png');plt.close(fig)
    # Same GT-only selected rays across the three checkpoints.
    example_keys=[(e['sample_id'],e['camera'],tuple(e['uv']),e['band']) for e in results[NAMES[0]]['examples'] if e['band']=='60-80m']
    if example_keys:
        fig,axes=plt.subplots(len(example_keys),1,figsize=(12,3.1*len(example_keys)),squeeze=False)
        from datasets.labels import depth_bin_edges
        edges=depth_bin_edges();centers=(edges[:-1]+edges[1:])/2
        for ax,key in zip(axes[:,0],example_keys):
            for n,label,color in zip(NAMES,SHORT,COLORS):
                e=next(e for e in results[n]['examples'] if (e['sample_id'],e['camera'],tuple(e['uv']),e['band'])==key)
                ax.plot(centers,e['probability'],label=f"{label}: mean={e['expected_m']:.1f}, mode={e['mode_m']:.1f}, H={e['entropy']:.2f}",color=color)
            ax.axvline(e['gt_m'],color='black',ls='--',label=f"GT={e['gt_m']:.1f}m; BEV-in={e['gt_in_bev']}")
            ax.set(xlim=(1.5,80),ylabel='Bin probability',title=f"{key[0]}, {key[1]}, UV={key[2]}");ax.legend(fontsize=8,ncol=2)
        axes[-1,0].set_xlabel('Optical depth bin center (m), unequal LID widths');fig.tight_layout();fig.savefig(root/'far_probability_examples.png');plt.close(fig)
    # Probability average is population mixture, explicitly not evidence of individual multimodality.
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,band in zip(axes,['1.5-10m','60-80m']):
        for n,label,color in zip(NAMES,SHORT,COLORS):ax.plot(centers,results[n]['average_probability_by_gt_band'][band]['probability'],label=label,color=color)
        ax.set(title=f'Population mean probability | GT {band}',xlabel='Depth center (m)',ylabel='Bin probability');ax.legend()
    fig.suptitle('Population mixtures do not establish per-ray multimodality');fig.tight_layout();fig.savefig(root/'mean_probability_profiles.png');plt.close(fig)
    conditions=['original',*report['perturbation_semantics']]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for i,(n,label,color) in enumerate(zip(NAMES,SHORT,COLORS)):
        r=results[n];depth=[r['distribution']['all_valid']['all']['all']['expected_mae_m']]+[r['interventions'][c]['all_valid']['all']['all']['expected_mae_m'] for c in conditions[1:]]
        road=[r['road_common_v1'][c]['overall']['iou_positive'] for c in conditions]
        axes[0].plot(range(len(conditions)),depth,'-o',label=label,color=color);axes[1].plot(range(len(conditions)),road,'-o',label=label,color=color)
    for ax in axes:ax.set(xticks=range(len(conditions)),xticklabels=conditions);ax.tick_params(axis='x',rotation=30);ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Expected MAE (m)');axes[1].set_ylabel('Road IoU, common v1 mask');fig.suptitle('Diagnostic interventions only; altered inputs are not formal model performance');fig.tight_layout();fig.savefig(root/'rgb_feature_interventions.png');plt.close(fig)
    agg={n:aggregate(results[n]) for n in NAMES};fig,axes=plt.subplots(1,2,figsize=(13,4))
    variants=['learned_distribution','oracle_gt_bin','expected','mode','median','pixel_prior']
    for n,label,color in zip(NAMES,SHORT,COLORS):
        axes[0].plot(range(len(variants)),[agg[n]['all']['variants'][v]['mass_frame_mean']['normalized_spatial_tv'] for v in variants],'-o',label=label,color=color)
        frames=results[n]['frames'];bf=results[n]['bev_frames'];axes[1].scatter([f['depth']['expected_mae_m'] for f in frames],[f['variants']['learned_distribution']['mass']['normalized_spatial_tv'] for f in bf],s=9,alpha=.4,label=label,color=color)
    axes[0].set(xticks=range(len(variants)),xticklabels=variants,ylabel='Mass spatial TV (lower better)');axes[0].tick_params(axis='x',rotation=25);axes[0].legend()
    axes[1].set(xlabel='Frame expected depth MAE (m)',ylabel='Matched-ray mass spatial TV');axes[1].legend();fig.tight_layout();fig.savefig(root/'depth_vs_bev_spatial_quality.png');plt.close(fig)
    for n in NAMES:
        for path in (Path(report['artifacts'])/n).glob('*_bev.npz'):
            with np.load(path,allow_pickle=False) as data:
                keys=['oracle_mass','oracle_gt_bin_mass','learned_distribution_mass','expected_mass','pixel_prior_mass']
                fig,axes=plt.subplots(len(keys),4,figsize=(12,13));vmax=max(float(np.log1p(data[k]).max()) for k in keys)
                for row,k in enumerate(keys):
                    for z in range(4):
                        im=axes[row,z].imshow(np.log1p(data[k][z,::-1,::-1]),vmin=0,vmax=vmax,cmap='magma',aspect='equal')
                        axes[row,z].set_xticks([]);axes[row,z].set_yticks([])
                        if row==0:axes[row,z].set_title(['z[-2,-.5)','z[-.5,1)','z[1,2.5)','z[2.5,4)'][z])
                        if z==0:axes[row,z].set_ylabel(k.replace('_mass',''),fontsize=9)
                fig.suptitle(f'{n} | {path.stem}\nShared log(1+count/mass) scale; sampled surfaces, NOT occupancy; forward up, left left',fontsize=10)
                fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='log(1+mass/count)');fig.savefig(root/(n+'_'+path.stem+'.png'),bbox_inches='tight');plt.close(fig)
    return agg
