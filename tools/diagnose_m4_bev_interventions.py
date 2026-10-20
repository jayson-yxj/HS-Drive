"""Supplementary full-ray depth/context interventions; preserve unseen-ray distributions."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from datasets.training import HSDriveDataset,make_dataloader
from training.runner import seed_all,initialize,load_checkpoint,to_device,road_metrics
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from diagnostics.experiment import RUNS,road_confusion,surface_metrics,read
from models.view_transform import feature_uv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m4_bev_full_ray_interventions.json')
    p.add_argument('--artifacts',default='reports/m4_bev_full_ray_artifacts')
    p.add_argument('--diagnostics',default='reports/m4_depth_distribution_diagnostics.json');a=p.parse_args()
    out=Path(a.output);root=Path(a.artifacts)
    if out.exists() or root.exists():raise FileExistsError('New outputs required')
    root.mkdir(parents=True);seed_all(7);prepare_cuda();results={}
    protected={str(p):file_hash(p) for folder in ('runs','reports') for p in Path(folder).rglob('*') if p.is_file()}
    main_report=read(a.diagnostics)
    if main_report['status']!='complete' or not all(main_report['engineering_checks'].values()):raise ValueError('Complete base diagnostics required')
    with np.load(Path(main_report['artifacts'])/'training_priors.npz') as d:prior=torch.from_numpy(d['camera_pixel_distribution']).permute(0,3,1,2)[None].cuda()
    gen=torch.Generator().manual_seed(91);permutation=torch.randperm(24*40,generator=gen).cuda()
    conditions=('original','depth_probability_shuffle','context_spatial_shuffle','oracle_gt_bin_valid_only','pixel_distribution_prior_valid_only')
    for name in RUNS:
        m=read(Path('runs')/name/'manifest.json');cfg=m['config'];model=initialize(cfg,pretrained=False).cuda().eval();load_checkpoint(Path('runs')/name/'best.pt',model)
        if file_hash(Path('runs')/name/'best.pt')!=main_report['results'][name]['checkpoint_sha256'] or file_hash(Path(cfg['index'])/'index.json')!=main_report['results'][name]['index_sha256']:raise ValueError('Base diagnostic identity mismatch')
        ds=HSDriveDataset(cfg['index'],'val');rows=[];conf={c:{} for c in conditions};shown=set();bound=.0
        with torch.inference_mode():
            for i,batch in enumerate(make_dataloader(ds,workers=2)):
                inputs,t=to_device(batch);sid=batch.metadata[0]['scene_id'];sample=batch.metadata[0]['sample_id'];valid=t['depth_valid'];counts=model.splat.observed_counts(t['depth_m'],t['depth_uv'],valid,inputs['intrinsics'],inputs['cam_to_ego'])
                fuv=feature_uv(24,40,inputs['images'].shape[-2:],'cuda');bound=max(bound,float((fuv-t['depth_uv']).abs().max()))
                outputs={}
                for condition in conditions:
                    def hook(module,args):
                        context,prob,k,extr,shape=args
                        if condition=='depth_probability_shuffle':prob=prob.flatten(-2)[...,permutation].reshape_as(prob)
                        elif condition=='context_spatial_shuffle':context=context.flatten(-2)[...,permutation].reshape_as(context)
                        elif condition=='oracle_gt_bin_valid_only':
                            one=F.one_hot(t['depth_bin'].clamp_min(0),48).permute(0,1,4,2,3).float();prob=torch.where(valid[:,:,None],one,prob)
                        elif condition=='pixel_distribution_prior_valid_only':prob=torch.where(valid[:,:,None],prior,prob)
                        return context,prob,k,extr,shape
                    handle=model.splat.register_forward_pre_hook(hook)
                    try:
                        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(**inputs.as_model_kwargs())
                    finally:handle.remove()
                    cm=road_confusion(o['drivable_logits'],t);conf[condition].setdefault(sid,np.zeros((2,2),np.int64))[:]+=cm
                    mass=o['splat_mass'];weak=(mass>1e-6)&(mass<1e-3);support=mass>1e-6
                    record={'road':road_metrics(cm),'surface_head':surface_metrics(o['surface_log_counts'],counts),
                        'splat_mass_total':float(mass.sum()),'nonzero_mass_voxels':int(support.sum()),'weak_mass_voxels':int(weak.sum()),
                        'weak_voxel_fraction':float(weak.sum()/support.sum().clamp_min(1)),
                        'weak_mass_fraction':float(mass[weak].sum()/mass.sum()),
                        'weak_feature_l1_fraction':float(o['height_bev'].float().abs().permute(0,2,3,4,1)[weak].sum()/o['height_bev'].float().abs().sum().clamp_min(1e-9))}
                    outputs[condition]=(o,record)
                clean=outputs['original'][0];row={'sample_id':sample,'scene':sid,'conditions':{}}
                for condition,(o,record) in outputs.items():
                    record['bev_relative_l1_change']=float((o['bev_current'].float()-clean['bev_current'].float()).abs().sum()/clean['bev_current'].float().abs().sum().clamp_min(1e-9))
                    record['road_probability_change_valid']=float((o['drivable_logits'].float().sigmoid()-clean['drivable_logits'].float().sigmoid()).abs()[t['drivable_valid']].mean())
                    row['conditions'][condition]=record
                rows.append(row)
                if sid not in shown:
                    shown.add(sid);prob=clean['depth_probs'];centers=model.splat.depth_centers
                    save={'rgb':(inputs['images'][0].cpu().numpy()*255).astype(np.uint8),'depth_gt':t['depth_m'][0].cpu().numpy(),'depth_valid':valid[0].cpu().numpy(),
                        'depth_expected':(prob*centers[None,None,:,None,None]).sum(2)[0].cpu().numpy(),'depth_mode':centers[prob.argmax(2)][0].cpu().numpy(),
                        'entropy':(-(prob*prob.clamp_min(1e-30).log()).sum(2)/np.log(48))[0].cpu().numpy(),'road_gt':t['drivable_static'][0].cpu().numpy(),'road_valid':t['drivable_valid'][0].cpu().numpy()}
                    for condition,(o,_) in outputs.items():save[condition+'_road']=o['drivable_logits'][0].float().sigmoid().cpu().numpy()
                    np.savez_compressed(root/(name+'_'+sid+'.npz'),**save)
                if i%60==0:print(name,i+1,len(ds),flush=True)
        results[name]={'frames':rows,'road':{c:{'overall':road_metrics(sum(s.values())),'per_scene':{sid:road_metrics(v) for sid,v in s.items()}} for c,s in conf.items()},
            'checkpoint_sha256':m['best_sha256'],'max_native_vs_supervision_uv_difference_pixels':bound}
        del model;torch.cuda.empty_cache()
    check=all(file_hash(p)==h for p,h in protected.items())
    atomic_json(out,{'status':'complete','old_artifacts_unchanged':check,'results':results,'conditions':list(conditions),'artifacts':str(root),
        'scope':'Fixed original RGB/context for depth interventions. Preserve native feature rays and all invalid-GT ray probabilities; replace ONLY valid labels. Oracle is privileged/OOD diagnostic, not performance upper bound.',
        'caveat':'GT is nearest native pixel mapped to feature ray; maximum UV discrepancy reported, <=0.5 resized pixels. No continuous Oracle claimed here.',
        'weak_mass_definition':'1e-6 < mass < 1e-3; existing height features normalized by max(mass,1e-6); thresholds diagnostic only',
        'protected_files_sha256':protected,'source_sha256':file_hash(__file__)})
    if not check:raise RuntimeError('Old files changed')

if __name__=='__main__':main()
