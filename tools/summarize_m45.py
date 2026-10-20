"""Reproducible offline M4.5 summaries and figures; never overwrite evidence."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets.occupancy_labels import validate_occupancy
from simulator.recorder import atomic_json,file_hash


def read(path):return json.loads(Path(path).read_text())


def road(cm):
    tn,fp,fn,tp=np.array(cm).ravel()
    return dict(iou=float(tp/max(tp+fp+fn,1)),precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1)),confusion=np.array(cm).tolist())


def mean(rows,key):return float(np.mean([r[key] for r in rows]))


def splat(prefix):
    source=read('reports/m45_splat_comparison.json');summary={}
    for name,result in source['results'].items():
        old=read(result['matched_ray_bev_evidence']);frames=result['frames'];groups={}
        assert [f['sample_id'] for f in frames]==old['sample_ids']
        for scene in ['all',*result['mass_quantiles']]:
            selected=[f for f in frames if scene=='all' or f['scene']==scene];modes={}
            oldframes=[f for f in old['bev_frames'] if scene=='all' or f['scene']==scene]
            for key in selected[0]['variants']:
                v=[f['variants'][key] for f in selected];buckets={}
                for group in v[0]['groups']:
                    vals=[r['groups'][group] for r in v];buckets[group]={k:sum(r[k] for r in vals) for k in vals[0]}
                total=sum(r['feature_l1_total'] for r in v)
                cm=sum((np.array(r['drivable']['confusion_tn_fp_fn_tp']) for r in v),np.zeros((2,2),int))
                shuffle_cm=sum((np.array(r['depth_shuffle_road']['confusion_tn_fp_fn_tp']) for r in v),np.zeros((2,2),int))
                modes[key]=dict(feature_l1_total=total,weak_mass_feature_fraction=sum(r['weak_mass_feature_fraction']*r['feature_l1_total'] for r in v)/total,
                    feature_fraction_without_sampled_gt=sum(b['feature_l1'] for k,b in buckets.items() if '/observed0/' in k)/total,
                    groups=buckets,road=road(cm),depth_shuffle_road=road(shuffle_cm),
                    **{k:mean(v,k) for k in ['BEV_relative_L1_change','BEV_feature_cosine_similarity','drivable_probability_change','depth_shuffle_probability_change']},
                    runtime_ms_p50=float(np.median([r['runtime_ms'] for r in v])),runtime_ms_p95=float(np.quantile([r['runtime_ms'] for r in v],.95)),
                    peak_cuda_allocated_mib=max(r['peak_cuda_allocated_mib'] for r in v),peak_cuda_reserved_mib=max(r['peak_cuda_reserved_mib'] for r in v))
            distributions=old['distribution']['gt_in_bev_3d']
            distribution=distributions['all']['all'] if scene=='all' else distributions['scene/'+scene]['all']
            groups[scene]=dict(frames=len(selected),mass_total_frame_mean=mean(selected,'mass_total'),mass_nonzero_voxels_frame_mean=mean(selected,'mass_nonzero_voxels'),
                native_mass_vs_sampled_oracle={k:mean([f['mass_vs_oracle'] for f in selected],k) for k in ['normalized_spatial_tv','mass_fraction_on_gt_support']},
                matched_valid_rays=dict(out_of_bev_gt_leakage_mass_frame_mean=mean(oldframes,'out_of_bev_gt_rays_probability_leaking_in'),
                    mass_on_exact_gt_voxel=distribution['mass_on_exact_gt_voxel'],mass_xy_within_1m_same_height=distribution['mass_xy_within_1m_same_height']),
                variants=modes)
        summary[name]=dict(groups=groups,mass_quantiles=result['mass_quantiles'],checkpoint_sha256=result['checkpoint_sha256'])
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        keys=list(groups['all']['variants']);x=np.arange(len(keys))
        for ax,field,title in zip(axes.flat[:3],['weak_mass_feature_fraction','feature_fraction_without_sampled_gt','depth_shuffle_probability_change'],['Weak-mass feature fraction','Feature fraction outside sampled GT support','Frozen road depth-shuffle response']):
            ax.bar(x,[groups['all']['variants'][k][field] for k in keys]);ax.set_xticks(x,keys,rotation=35,ha='right');ax.set_title(title)
        npz=Path('reports/m45_splat_figures')
        for ax,key in zip(axes.flat[3:],['baseline','sum','gated_0.01']):
            with np.load(npz/f'{name}_0_{key}.npz') as arrays:
                im=ax.imshow(np.log1p(arrays['feature_l1'].sum(0))[::-1,::-1],cmap='magma',vmin=0,vmax=7)
            ax.set_title(key+' log1p feature L1');ax.set_xlabel('Y right to left');ax.set_ylabel('X front to rear')
        fig.colorbar(im,ax=list(axes.flat[3:]),shrink=.7);fig.suptitle(name+' | frozen-input diagnostic, not trained ranking')
        fig.subplots_adjust(bottom=.08,top=.9,hspace=.65,wspace=.35);fig.savefig(prefix.parent/(prefix.name+'_'+name+'_splat.png'));plt.close(fig)
        # Identical mass for all aggregators: show Oracle support vs learned across all heights.
        with np.load(npz/f'{name}_0_baseline.npz') as arrays:
            fig,axes=plt.subplots(2,4,figsize=(12,7))
            vmax=max(float(np.log1p(arrays['mass']).max()),float(np.log1p(arrays['oracle']).max()))
            for z in range(4):
                axes[0,z].imshow(np.log1p(arrays['oracle'][z])[::-1,::-1],vmin=0,vmax=vmax);axes[0,z].set_title(f'Oracle count z{z}')
                axes[1,z].imshow(np.log1p(arrays['mass'][z])[::-1,::-1],vmin=0,vmax=vmax);axes[1,z].set_title(f'Native learned mass z{z}')
            fig.suptitle('Native full-ray vs sampled GT: coverage differs; not occupancy');fig.tight_layout();fig.savefig(prefix.parent/(prefix.name+'_'+name+'_mass.png'));plt.close(fig)
    atomic_json(str(prefix)+'_analysis_metrics.json',dict(results=summary,aggregation='Continuous metrics frame mean; feature fractions pooled L1; road pooled confusion. Geometry invariant under A/B/C; old matched-ray results reused with sample/hash checks.',source_sha256=file_hash('reports/m45_splat_comparison.json')))


def occupancy(prefix):
    records=[];root=prefix.parent/(prefix.name+'_occupancy_figures');root.mkdir()
    base=Path('/root/gpufree-data/hs-drive-data/processed')
    for scene in ['m45_occupancy_legacy_town04','m45_occupancy_clear','m45_occupancy_fog']:
        manifest=read(base/scene/'manifest.json');stats=[]
        shutil.copy2(base/scene/'manifest.json',root/(scene+'_manifest.json'))
        for index,row in enumerate(manifest['records']):
            path=base/scene/row['path'];assert file_hash(path)==row['sha256']
            with np.load(path,allow_pickle=False) as arr:labels={k:arr[k] for k in arr.files}
            validate_occupancy(labels)
            state=labels['occupancy_state'];layers=[]
            for z in range(4):
                layers.append(dict(free=int((state[z]==1).sum()),surface=int((state[z]==2).sum()),unknown=int((state[z]==0).sum()),ray_conflicts=int(labels['ray_conflict_mask'][z].sum()),ground=int((labels['surface_semantic'][z]==1).sum()),obstacle=int(labels['obstacle_mask'][z].sum())))
            stats.append(dict(**row,layers=layers,all_invariants_pass=True,unknown_not_negative=bool(np.all(~labels['occupancy_valid'][state==0]))))
            if index==0:
                shutil.copy2(path,root/(scene+'_example.npz'));fig,axes=plt.subplots(3,4,figsize=(13,9))
                for z in range(4):
                    axes[0,z].imshow(state[z][::-1,::-1],cmap=ListedColormap(['#333333','#70c1b3','#e76f51']),vmin=0,vmax=2);axes[0,z].set_title(f'z{z} unknown/free/surface')
                    axes[1,z].imshow(labels['surface_semantic'][z][::-1,::-1],cmap=ListedColormap(['#333333','#d6cb67','#6290c8','#d9598c','#aaaaaa']),vmin=0,vmax=4);axes[1,z].set_title('Ground/static/dynamic')
                    axes[2,z].imshow(np.log1p(labels['surface_count'][z])[::-1,::-1],cmap='magma');axes[2,z].set_title('Surface Count (not free)')
                for ax in axes.flat:ax.set_xlabel('Y right to left');ax.set_ylabel('X front to rear')
                fig.suptitle(scene+' | observed rays only; unobserved cells remain Unknown');fig.tight_layout();fig.savefig(root/(scene+'.png'));plt.close(fig)
        records.append(dict(scene=scene,source=manifest['source'],semantic_available=manifest['semantic_available'],manifest_sha256=file_hash(base/scene/'manifest.json'),records=stats))
    atomic_json(str(prefix)+'_occupancy_validation.json',dict(schema_version='hsd-m45-occupancy-validation-v1',geometric_invariants_passed=True,training_quality_gate='Partial',scenes=records,
        limitations=['Only 9 labeled frames, 6 with semantics; paired weather is same fixed pose','Self bounding box conservatively rejects rays; near-ground coverage limited','Voxel Free means ray-observed free evidence, not proof the entire coarse voxel is empty','Semantic conflicts are invalid semantic/obstacle supervision; geometric reliable surface evidence retained','Unknown/other classes are not obstacle negatives; no motion truth or traffic permission'] ))


def pairs(prefix):
    base=Path('/root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs');paths=[base/'clear',base/'fog'];frames=[sorted((p/'frames').glob('*.json')) for p in paths];rows=[]
    for left,right in zip(*frames):
        a,b=read(left),read(right);poses=float(np.max(np.abs(np.array(a['T_world_from_ego'])-np.array(b['T_world_from_ego']))));cameras={}
        for camera in a['cameras']:
            refs=[a['cameras'][camera],b['cameras'][camera]]
            for parent,ref in zip(paths,refs):
                for kind in ['rgb','depth','semantic']:assert file_hash(parent/ref[kind])==ref[kind+'_sha256']
            rgb=[cv2.imread(str(parent/ref['rgb'])) for parent,ref in zip(paths,refs)]
            dep=[np.load(parent/ref['depth'])['depth_m'] for parent,ref in zip(paths,refs)]
            sem=[np.load(parent/ref['semantic'])['semantic'] for parent,ref in zip(paths,refs)]
            cameras[camera]=dict(rgb_mae_255=float(np.abs(rgb[0].astype(float)-rgb[1]).mean()),depth_abs_mean_m=float(np.abs(dep[0]-dep[1]).mean()),depth_abs_max_m=float(np.abs(dep[0]-dep[1]).max()),semantic_changed_fraction=float((sem[0]!=sem[1]).mean()))
        rows.append(dict(frames=[a['frame_id'],b['frame_id']],pose_max_abs_error=poses,cameras=cameras))
    fig,axes=plt.subplots(2,6,figsize=(16,5))
    for row,(parent,files) in enumerate(zip(paths,frames)):
        f=read(files[0])
        for col,(camera,ref) in enumerate(f['cameras'].items()):
            axes[row,col].imshow(cv2.cvtColor(cv2.imread(str(parent/ref['rgb'])),cv2.COLOR_BGR2RGB));axes[row,col].set_title(parent.name+'/'+camera);axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(str(prefix)+'_weather_pairs.png');plt.close(fig)
    atomic_json(str(prefix)+'_paired_validation.json',dict(schema_version='hsd-m45-pair-v1',source=read(base/'pair_manifest.json'),rows=rows,all_poses_exact=all(r['pose_max_abs_error']==0 for r in rows),scope='Paired rendering diagnostic only; no trained quality claim. Small GT changes measured, not assumed identical. No motion-paired data collected.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='reports/m45');a=p.parse_args();prefix=Path(a.prefix)
    for suffix in ['_analysis_metrics.json','_occupancy_validation.json','_paired_validation.json']:
        if Path(str(prefix)+suffix).exists():raise FileExistsError(str(prefix)+suffix)
    splat(prefix);occupancy(prefix);pairs(prefix)


if __name__=='__main__':main()
