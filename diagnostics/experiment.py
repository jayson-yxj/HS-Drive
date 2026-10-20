"""Fixed-checkpoint evaluation and interventions; no optimizers or weight updates."""
import hashlib
import json
import time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from datasets.training import HSDriveDataset,make_dataloader
from datasets.schema import CAMERA_IDS
from datasets.labels import depth_bin_edges
from models.view_transform import unproject
from training.runner import initialize,load_checkpoint,seed_all,to_device,road_metrics
from training.losses import logits_at_target_uv
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from diagnostics.distributions import distribution_metrics,GroupedMetrics,BANDS,js_divergence
from diagnostics.spatial import (ray_geometry,project,downstream,compare_mass,compare_features,spatial_probability_metrics)

RUNS=('m4_expanded_joint','m4_far_balanced','m4_road_point')
CONDITIONS=('front_local_mask','rgb_spatial_mean','feature_spatial_shuffle','feature_zero')


def read(path):return json.loads(Path(path).read_text())


def fit_priors(index,chosen,output):
    """Fit ONLY the historical training sample IDs; exact continuous camera/pixel mean."""
    ds=HSDriveDataset(index,'train');ids=[i for i,w in enumerate(ds.windows) if w[-1]['sample_id'] in chosen]
    if {ds.windows[i][-1]['sample_id'] for i in ids}!=set(chosen):raise ValueError('Prior subset mismatch')
    hist=torch.zeros(6,24,40,48,dtype=torch.float64);sums=torch.zeros(6,24,40,dtype=torch.float64);counts=torch.zeros_like(sums)
    road_pos=torch.zeros(160,120,dtype=torch.float64);road_n=torch.zeros_like(road_pos);band_counts={k:0 for k in BANDS if k!='all'};band_per_frame=[]
    for batch in make_dataloader(torch.utils.data.Subset(ds,ids),workers=2):
        t=batch.targets;valid=t['depth_valid'][0];depth=t['depth_m'][0];labels=t['depth_bin'][0]
        hist+=F.one_hot(labels.clamp_min(0),48)*valid[...,None];sums+=depth.double()*valid;counts+=valid
        v=t['drivable_valid'][0];road_pos+=(t['drivable_static'][0]==1)&v;road_n+=v
        frame_counts={k:int((valid&(depth>=lo)&(depth<hi)).sum()) for k,(lo,hi) in BANDS.items() if k!='all'}
        for k,vv in frame_counts.items():band_counts[k]+=vv
        band_per_frame.append(frame_counts)
    centers=torch.tensor((depth_bin_edges()[:-1]+depth_bin_edges()[1:])/2)
    global_hist=hist.sum((0,1,2));global_mean=float((global_hist*centers).sum()/global_hist.sum())
    mean=torch.where(counts>0,sums/counts.clamp_min(1),global_mean).float()
    probability=torch.where(counts[...,None]>0,hist/counts.clamp_min(1)[...,None],global_hist/global_hist.sum()).float()
    global_road=float(road_pos.sum()/road_n.sum());road=torch.where(road_n>0,road_pos/road_n.clamp_min(1),global_road).float()
    np.savez_compressed(output,camera_pixel_mean=mean.numpy(),camera_pixel_distribution=probability.numpy(),road_probability=road.numpy(),counts=counts.numpy(),histogram=hist.numpy())
    frequencies={k:sum(v[k]/max(sum(v.values()),1) for v in band_per_frame)/len(band_per_frame) for k in band_counts}
    equal={k:sum((1/sum(c>0 for c in v.values())) if v[k] else 0 for v in band_per_frame)/len(band_per_frame) for k in band_counts}
    return mean,probability,road,{'train_samples':len(ids),'train_sample_ids':chosen,'band_pixel_counts':band_counts,
        'average_unbalanced_band_weight_per_frame':frequencies,'average_equal_band_weight_per_frame':equal,
        'effective_rare_band_upweight_ratio':{k:equal[k]/frequencies[k] if frequencies[k] else None for k in equal},
        'missing_pixel_cells':int((counts==0).sum()),'global_depth_fallback_m':global_mean,'road_global_fallback':global_road,
        'scope':'Actual 299 training samples only. Histogram prior is empirical, not fitted on validation. Road prior uses common legacy v1 labels.'}


@contextmanager
def replace_encoder(model,replacement):
    handle=model.encoder.register_forward_hook(lambda module,args,output:replacement)
    try:yield
    finally:handle.remove()


def input_variant(images,condition):
    if condition=='rgb_spatial_mean':return images.mean((-1,-2),keepdim=True).expand_as(images).clone()
    if condition=='front_local_mask':
        result=images.clone();h,w=images.shape[-2:]
        result[:,0,:,h//4:3*h//4,w//4:3*w//4]=images[:,0].mean((-1,-2),keepdim=True)
        return result
    raise ValueError(condition)


def sample_context(context,uv,image_hw):
    # Same bilinear convention as depth logits; geometry uses these exact rays.
    return logits_at_target_uv(context,uv,image_hw).permute(0,1,3,4,2)


def road_confusion(logits,target):
    m=target['drivable_valid'];gt=target['drivable_static'].long();pred=logits>0
    return torch.bincount(gt[m]*2+pred[m].long(),minlength=4).reshape(2,2).cpu().numpy()


def flat_probs(output,target,inputs):
    p=logits_at_target_uv(output['depth_logits'],target['depth_uv'],inputs['images'].shape[-2:]).softmax(2)
    return p.permute(0,1,3,4,2)[target['depth_valid']]


def scalar_means(values,mask=None):
    if mask is None:mask=np.ones(len(next(iter(values.values()))),bool)
    return {'count':int(mask.sum()),**{k:float(v[mask].mean(dtype=np.float64)) if mask.any() else None for k,v in values.items()}}


def surface_metrics(pred,oracle):
    error=(pred.float()-torch.log1p(oracle)).abs();positive=oracle>0
    return {'mae_all':float(error.mean()),'mae_positive':float(error[positive].mean()) if positive.any() else None,
            'mae_zero_sample_cells':float(error[~positive].mean()),'negative_prediction_fraction':float((pred<0).float().mean())}


def evaluate_checkpoint(name,root,man,prior_values,limit=None):
    config=man['config'];seed_all(7);model=initialize(config,pretrained=False).cuda().eval()
    state=load_checkpoint(Path('runs')/name/'best.pt',model)
    if state['config']!=config or state['index_sha256']!=man['index_sha256']:raise ValueError('Checkpoint identity mismatch')
    ds=HSDriveDataset(config['index'],'val') # common v1 road labels for all models
    if limit:ds=torch.utils.data.Subset(ds,list(range(min(limit,len(ds)))))
    centers=model.splat.depth_centers;edges=torch.tensor(depth_bin_edges(),device='cuda',dtype=torch.float32)
    pixel_mean,prior_p,road_prior=prior_values
    pixel_mean=pixel_mean.cuda();prior_p=prior_p.cuda();road_prior=road_prior.cuda()
    grouped=GroupedMetrics();prior_grouped=GroupedMetrics();interventions={c:GroupedMetrics() for c in CONDITIONS}
    confusion={c:np.zeros((2,2),np.int64) for c in ('original',*CONDITIONS,'position_prior')};scene_conf={c:{} for c in confusion}
    frame_rows=[];bev_rows=[];examples=[];sample_ids=[];fingerprint=hashlib.sha256();selected_examples=set();arrays_dir=root/name;arrays_dir.mkdir()
    probability_by_band={k:np.zeros(48) for k in BANDS};counts_by_band={k:0 for k in BANDS}
    generator=torch.Generator().manual_seed(91);permutation=torch.randperm(24*40,generator=generator).cuda()
    start=time.monotonic()
    with torch.inference_mode():
        for frame_no,batch in enumerate(make_dataloader(ds,workers=2)):
            inputs,target=to_device(batch);kwargs=inputs.as_model_kwargs();meta=batch.metadata[0];sid=meta['scene_id'];sample_id=meta['sample_id'];sample_ids.append(sample_id)
            valid=target['depth_valid'];depth=target['depth_m'];uv=target['depth_uv'];speed=float(inputs['ego_state'][0,0]);motion='moving_gt_speed_gt1' if speed>1 else 'stationary_or_slow_le1'
            for value in (depth,uv,valid):fingerprint.update(value.cpu().numpy().tobytes())
            captured={}
            handles=[model.encoder.register_forward_hook(lambda m,a,o:captured.update(features=o)),model.context_head.register_forward_hook(lambda m,a,o:captured.update(context=o))]
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):original=model(**kwargs)
            finally:
                for handle in handles:handle.remove()
            p=flat_probs(original,target,inputs);gt=depth[valid];bins=target['depth_bin'][valid]
            pts,rank,inside,xy=ray_geometry(model.splat,uv,depth,inputs['intrinsics'],inputs['cam_to_ego'])
            gt_pts=pts[valid];gt_rank=rank[valid];gt_in=inside[valid];xy_in=xy[valid]
            camera=np.array(CAMERA_IDS)[np.broadcast_to(np.arange(6)[None,:,None,None],tuple(valid.shape))[valid.cpu().numpy()]]
            args=dict(camera=camera,scene=sid,motion=motion,in_bev=gt_in.cpu().numpy(),xy_in_bev=xy_in.cpu().numpy())
            vals=distribution_metrics(p,centers,gt,bins,edges)
            # Matched exact rays, optical_z -> HS ego. This differs from full native model ray coverage.
            d=centers.view(1,1,48,1,1).expand(1,6,48,24,40)
            bin_points=unproject(uv[:,:,None].expand(1,6,48,24,40,2),d,inputs['intrinsics'],inputs['cam_to_ego']).permute(0,1,3,4,2,5)[valid]
            vals.update(spatial_probability_metrics(p,bin_points,gt_pts,gt_rank,gt_in,model.splat))
            grouped.update(vals,**args)
            for band,(low,high) in BANDS.items():
                mask=(gt>=low)&(gt<high);probability_by_band[band]+=p[mask].sum(0).cpu().numpy();counts_by_band[band]+=int(mask.sum())
            prior=prior_p[None].expand(1,-1,-1,-1,-1)[valid];pv=distribution_metrics(prior,centers,gt,bins,edges)
            continuous=pixel_mean[None][valid];pv['continuous_pixel_mean_mae_m']=(continuous-gt).abs().cpu().numpy();pv['continuous_pixel_mean_signed_bias_m']=(continuous-gt).cpu().numpy()
            prior_grouped.update(pv,**args)
            base_road=road_confusion(original['drivable_logits'],target);confusion['original']+=base_road
            pp_logits=(road_prior[None]-.5)*2;pm=road_confusion(pp_logits,target);confusion['position_prior']+=pm
            for c,m in [('original',base_road),('position_prior',pm)]:scene_conf[c].setdefault(sid,np.zeros((2,2),np.int64))[:]+=m
            frame_rows.append({'sample_id':sample_id,'scene':sid,'speed_mps':speed,'motion':motion,'depth':scalar_means(vals),'gt_in_bev':scalar_means(vals,args['in_bev']),
                               'road':road_metrics(base_road),'prior_mae_m':float((continuous-gt).abs().mean())})
            perturb_rows={}
            for condition in CONDITIONS:
                kw=dict(kwargs)
                if condition.startswith('feature_'):
                    features=captured['features'];replacement=torch.zeros_like(features) if condition=='feature_zero' else features.flatten(-2)[...,permutation].reshape_as(features)
                    with replace_encoder(model,replacement),torch.autocast('cuda',dtype=torch.bfloat16):altered=model(**kw)
                else:
                    kw['images']=input_variant(inputs['images'],condition)
                    with torch.autocast('cuda',dtype=torch.bfloat16):altered=model(**kw)
                ap=flat_probs(altered,target,inputs);av=distribution_metrics(ap,centers,gt,bins,edges)
                av['probability_js_divergence_nats']=js_divergence(p,ap).cpu().numpy()
                av['expected_depth_change_m']=((ap*centers).sum(-1)-(p*centers).sum(-1)).abs().cpu().numpy()
                av['expected_mae_delta_vs_original_m']=av['expected_mae_m']-vals['expected_mae_m']
                interventions[condition].update(av,**args)
                cm=road_confusion(altered['drivable_logits'],target);confusion[condition]+=cm;scene_conf[condition].setdefault(sid,np.zeros((2,2),np.int64))[:]+=cm
                road_mask=target['drivable_valid'];road_delta=(altered['drivable_logits'].float().sigmoid()-original['drivable_logits'].float().sigmoid()).abs()
                perturb_rows[condition]={'depth':scalar_means(av),'road':road_metrics(cm),'road_probability_change_valid':float(road_delta[road_mask].mean()),
                    'road_flip_fraction_valid':float(((altered['drivable_logits']>0)!=(original['drivable_logits']>0))[road_mask].float().mean()),
                    'bev_relative_l1_change':float((altered['bev_current'].float()-original['bev_current'].float()).abs().sum()/original['bev_current'].float().abs().sum().clamp_min(1e-9))}
                if condition=='front_local_mask':
                    flat_uv=uv[valid];local=(camera=='front')&(flat_uv[:,0].cpu().numpy()>=160)&(flat_uv[:,0].cpu().numpy()<480)&(flat_uv[:,1].cpu().numpy()>=96)&(flat_uv[:,1].cpu().numpy()<288)
                    perturb_rows[condition]['masked_front_rays']=scalar_means(av,local);perturb_rows[condition]['unmasked_rays']=scalar_means(av,~local)
            frame_rows[-1]['interventions']=perturb_rows
            # Freeze context, feature sampling, masks and downstream heads; vary only depth.
            context=captured['context'].reshape(1,6,64,24,40);ctx=sample_context(context,uv,inputs['images'].shape[-2:])[valid]
            oracle_h,oracle_m=project(ctx,torch.ones(len(gt),1,device='cuda'),gt_pts[:,None],model.splat)
            counts=model.splat.observed_counts(depth,uv,valid,inputs['intrinsics'],inputs['cam_to_ego'])
            if not torch.equal(oracle_m,counts):raise ValueError('Oracle splat mismatches existing Surface Count interface')
            variants={'learned_distribution':(p,bin_points),'oracle_continuous':(torch.ones(len(gt),1,device='cuda'),gt_pts[:,None]),
                      'oracle_gt_bin':(torch.ones(len(gt),1,device='cuda'),bin_points[torch.arange(len(gt),device='cuda'),bins][:,None])}
            # Along the same ray: derive point at any depth without altering geometry.
            cam_idx=torch.as_tensor([CAMERA_IDS.index(c) for c in camera],device='cuda')
            origins=inputs['cam_to_ego'][0,cam_idx,:3,3].float();rays=(gt_pts-origins)/gt[:,None]
            for kind,est in [('expected',(p*centers).sum(-1)),('mode',centers[p.argmax(-1)]),('median',centers[(p.cumsum(-1)>=.5).long().argmax(-1)]),('pixel_prior',continuous)]:
                variants[kind]=(torch.ones(len(gt),1,device='cuda'),(origins+rays*est[:,None])[:,None])
            # Actual-depth-conditioned probability transport including out-of-BEV GT rays.
            mask_out=~gt_in;leak_mass=float((p[mask_out]*model.splat.ranks(bin_points[mask_out])[1]).sum());gt_count=int(gt_in.sum())
            bev_record={'sample_id':sample_id,'scene':sid,'gt_in_bev_rays':gt_count,'valid_rays':len(gt),
                'out_of_bev_gt_rays_probability_leaking_in':leak_mass,'native_surface_head':surface_metrics(original['surface_log_counts'],counts),
                'native_surface_head_road':road_metrics(base_road),'variants':{}}
            stored={'oracle_mass':oracle_m[0].cpu().numpy(),'native_surface_log_counts':original['surface_log_counts'][0].float().cpu().numpy()}
            for kind,(weights,points) in variants.items():
                h,m=project(ctx,weights,points,model.splat)
                with torch.autocast('cuda',dtype=torch.bfloat16):down=downstream(model,h,m)
                cm=road_confusion(down['road'],target)
                bev_record['variants'][kind]={'mass':compare_mass(m,oracle_m),'features':compare_features(h,oracle_h,m,oracle_m),
                    'surface_head':surface_metrics(down['surface'],counts),'road':road_metrics(cm)}
                stored[kind+'_mass']=m[0].cpu().numpy()
            bev_rows.append(bev_record)
            # Fixed first frame per scene: probability examples chosen by GT only, never prediction quality.
            if sid not in selected_examples:
                selected_examples.add(sid)
                stored.update(rgb=(inputs['images'][0].cpu().numpy()*255).astype(np.uint8),road_gt=target['drivable_static'][0].cpu().numpy(),road_valid=target['drivable_valid'][0].cpu().numpy(),road_pred=original['drivable_logits'][0].float().sigmoid().cpu().numpy())
                np.savez_compressed(arrays_dir/(sid+'_bev.npz'),**stored)
                for band,(low,high) in BANDS.items():
                    if band=='all':continue
                    ids=np.flatnonzero((vals['gt_depth_m']>=low)&(vals['gt_depth_m']<high))
                    if len(ids):
                        for idx in np.unique(ids[np.linspace(0,len(ids)-1,min(3,len(ids)),dtype=int)]):
                            examples.append({'sample_id':sample_id,'camera':str(camera[idx]),'uv':uv[valid][idx].cpu().tolist(),'band':band,'gt_m':float(gt[idx]),'gt_in_bev':bool(gt_in[idx]),
                                'probability':p[idx].cpu().tolist(),'expected_m':float((p[idx]*centers).sum()),'mode_m':float(centers[p[idx].argmax()]),'entropy':float(vals['normalized_entropy'][idx])})
            if frame_no%20==0:print(name,'frame',frame_no+1,'/',len(ds),'elapsed',round(time.monotonic()-start,1),flush=True)
    # Use identical full validation target fingerprint across checkpoints.
    result={'run':name,'checkpoint_sha256':man['best_sha256'],'selected_step':man['best_step'],'index_sha256':man['index_sha256'],
        'sample_ids':sample_ids,'target_depth_uv_valid_sha256':fingerprint.hexdigest(),'distribution':grouped.result(),'camera_pixel_prior':prior_grouped.result(),
        'interventions':{c:v.result() for c,v in interventions.items()},
        'road_common_v1':{c:{'overall':road_metrics(m),'per_scene':{s:road_metrics(v) for s,v in scene_conf[c].items()}} for c,m in confusion.items()},
        'frames':frame_rows,'bev_frames':bev_rows,'examples':examples,
        'average_probability_by_gt_band':{k:{'count':counts_by_band[k],'probability':(p/max(1,counts_by_band[k])).tolist()} for k,p in probability_by_band.items()},
        'wall_s':time.monotonic()-start,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    atomic_json(root/(name+'.json'),result)
    del model;torch.cuda.empty_cache()
    return result
