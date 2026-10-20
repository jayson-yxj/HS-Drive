"""Evaluate a validation-selected checkpoint on an entire held-out split."""
import argparse,json
from pathlib import Path
import torch
from datasets.training import HSDriveDataset,make_dataloader
from simulator.recorder import atomic_json,file_hash
from training.runner import seed_all,initialize,load_checkpoint,evaluate,write_preview,to_device
from training.metrics import DepthMetrics
from training.runtime import prepare_cuda


@torch.no_grad()
def training_depth_prior(train,train_sample_ids,centers):
    selected=set(train_sample_ids);counts=torch.zeros(len(centers),dtype=torch.float64)
    pixel_sum=torch.zeros(6,24,40,dtype=torch.float64);pixel_count=torch.zeros_like(pixel_sum)
    for i,w in enumerate(train.windows):
        if w[-1]['sample_id'] not in selected:continue
        target=train[i].targets;valid=target['depth_valid'];labels=target['depth_bin'][valid]
        counts+=torch.bincount(labels,minlength=len(centers))
        pixel_sum+=torch.where(valid,target['depth_m'].double(),0)
        pixel_count+=valid
    if counts.sum()==0:raise ValueError('Empty training prior')
    constant=float((counts*centers.cpu().double()).sum()/counts.sum())
    pixel_mean=torch.where(pixel_count>0,pixel_sum/pixel_count.clamp_min(1),constant).float()
    return constant,counts.tolist(),pixel_mean,int((pixel_count==0).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Run is not complete')
    if file_hash(run/'best.pt')!=manifest['best_sha256']:raise ValueError('Checkpoint checksum mismatch')
    config=manifest['config'];seed_all(config['seed']);prepare_cuda()
    if file_hash(Path(config['index'])/'index.json')!=manifest['index_sha256']:raise ValueError('Dataset changed')
    if file_hash(config['bev_config'])!=manifest['source_sha256'][config['bev_config']]:raise ValueError('BEV geometry changed')
    if config.get('road_overlay') and file_hash(Path(config['road_overlay'])/'manifest.json')!=manifest.get('road_overlay_manifest_sha256'):raise ValueError('Road overlay changed')
    model=initialize(config,pretrained=False).cuda();state=load_checkpoint(run/'best.pt',model)
    if state['config']!=config:raise ValueError('Checkpoint/run configuration mismatch')
    dataset=HSDriveDataset(config['index'],a.split,road_overlay=config.get('road_overlay'))
    if config.get('drivable_head') and dataset.index.get('drivable_bev_config_sha256')!=file_hash(config['bev_config']):
        raise ValueError('Static lane labels use different BEV geometry')
    report,preview=evaluate(model,dataset,list(range(len(dataset))),config,workers=config['workers'])
    write_preview(preview,root,a.split)
    # This deliberately weak constant-depth comparison is fit only on the actual training subset.
    train=HSDriveDataset(config['index'],'train',road_overlay=config.get('road_overlay'))
    prior,counts,pixel_mean,missing=training_depth_prior(train,manifest['train_sample_ids'],model.splat.depth_centers)
    metrics=DepthMetrics();pixel_metrics=DepthMetrics();pixel_scenes={}
    for batch in make_dataloader(dataset,workers=config['workers']):
        target=batch.targets
        pixel_metrics.update(pixel_mean[None].expand_as(target['depth_m']),target['depth_m'],target['depth_valid'])
        for i,meta in enumerate(batch.metadata):
            pixel_scenes.setdefault(meta['scene_id'],DepthMetrics()).update(pixel_mean,target['depth_m'][i],target['depth_valid'][i])
        metrics.update(torch.full_like(target['depth_m'],prior),target['depth_m'],target['depth_valid'])
    # Reloaded RGB-only path must reproduce the same outputs without any target I/O.
    inference=HSDriveDataset(config['index'],a.split,supervision=False)
    from unittest.mock import patch
    model.eval()
    with patch('numpy.load',side_effect=AssertionError('RGB inference opened Depth')):
        batch=next(iter(make_dataloader(inference,workers=0)))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp_bf16']):
            inputs=batch.inputs.to('cuda');output=model(**inputs.as_model_kwargs())
    road=report['drivable_static']
    matrix=torch.tensor(road['confusion_tn_fp_fn_tp'])
    positives=int(matrix[1].sum());negative=int(matrix[0].sum())
    road['all_positive_prior_iou']=positives/(positives+negative) if positives+negative else None
    report.update(schema_version='hsd-m4-eval-v1',split=a.split,checkpoint_sha256=manifest['best_sha256'],
        index_sha256=manifest['index_sha256'],selected_step=state['step'],model_selection='minimum validation '+config.get('selection_metric','total_loss')+'; no test used',
        prior_constant_m=prior,training_bin_histogram=counts,constant_prior_depth=metrics.result(),camera_pixel_prior_depth=pixel_metrics.result(),
        camera_pixel_prior_per_scene={k:v.result() for k,v in pixel_scenes.items()},camera_pixel_prior_missing_cells=missing,camera_pixel_prior_semantics='Per-camera per-feature-pixel mean training depth; fallback to global bin-center mean; training subset only',
        inference_without_supervision=True,output_shapes={k:list(v.shape) for k,v in output.items()},
        scope='Sampled valid smooth surface pixels only; tiny scene set, no driving/occupancy accuracy claim')
    atomic_json(root/'metrics.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
