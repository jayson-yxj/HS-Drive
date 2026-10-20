"""Small-data experiments with deterministic sample plans and atomic checkpoints."""
import json,os,random
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from datasets.training import HSDriveDataset,make_dataloader
from datasets.inputs import LearnedInputs
from models.depth_bev import DepthBEV
from simulator.recorder import atomic_json,file_hash
from training.losses import compute_losses,loss_options
from training.metrics import DepthMetrics,depth_prediction


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def select_indices(dataset,per_scene):
    if per_scene<1:raise ValueError('Need at least one sample per scene')
    groups={}
    for i,w in enumerate(dataset.windows):groups.setdefault(w[-1]['scene_id'],[]).append(i)
    indices=[]
    for group in groups.values():
        indices.extend(group[i] for i in np.unique(np.linspace(0,len(group)-1,min(per_scene,len(group)),dtype=int)))
    return indices


def to_device(batch,device='cuda'):
    inputs=batch.inputs.to(device)
    targets={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.targets.items()}
    return inputs,targets


def initialize(config,*,pretrained=True):
    bev=yaml.safe_load(Path(config['bev_config']).read_text())
    return DepthBEV(bev,weights_path=config['pretrained_weights'] if pretrained else None,
                    chunk_points=config['chunk_points'],drivable=config.get('drivable_head',False),road_context=config.get('road_context',False))


def atomic_checkpoint(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path,model,optimizer,step,config,index_sha,best_score):
    atomic_checkpoint(path,{'schema_version':'hsd-checkpoint-v1','model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'step':step,'config':config,'index_sha256':index_sha,
        'best_score':best_score,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})


def load_checkpoint(path,model,optimizer=None):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['schema_version']!='hsd-checkpoint-v1':raise ValueError('Unknown checkpoint')
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def loss_average(totals,count):return {k:v/count for k,v in totals.items()}


@torch.no_grad()
def evaluate(model,dataset,indices,config,*,workers=0):
    model.eval();totals={k:0. for k in ('total','depth_ce','surface_mse','depth_log','drivable_bce_dice')};metrics=DepthMetrics();quantized=DepthMetrics()
    sample_ids=[];surface_sum=0.;surface_cells=0;first=None
    confusion=np.zeros((2,2),dtype=np.int64);valid_cells=0;all_cells=0;scene_metrics={};road_scenes={};junction_confusion=np.zeros((2,2),dtype=np.int64)
    subset=torch.utils.data.Subset(dataset,indices)
    for batch in make_dataloader(subset,workers=workers,shuffle=False):
        inputs,targets=to_device(batch)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp_bf16']):
            output=model(**inputs.as_model_kwargs())
            losses,counts=compute_losses(model,output,inputs,targets,**loss_options(config))
        for k in totals:totals[k]+=float(losses[k])
        prediction=depth_prediction(output,targets,inputs,model.splat.depth_centers)
        metrics.update(prediction,targets['depth_m'],targets['depth_valid'])
        for i,meta in enumerate(batch.metadata):
            scene_metrics.setdefault(meta['scene_id'],DepthMetrics()).update(prediction[i],targets['depth_m'][i],targets['depth_valid'][i])
        q=model.splat.depth_centers[targets['depth_bin'].clamp_min(0)]
        quantized.update(q,targets['depth_m'],targets['depth_valid'])
        surface_sum+=float((output['surface_log_counts'].float()-torch.log1p(counts)).abs().sum())
        surface_cells+=counts.numel();sample_ids.extend(m['sample_id'] for m in batch.metadata)
        if 'drivable_logits' in output:
            mask=targets['drivable_valid'];pred=output['drivable_logits']>0;truth=targets['drivable_static'].long()
            confusion+=torch.bincount((truth[mask]*2+pred[mask].long()),minlength=4).reshape(2,2).cpu().numpy()
            valid_cells+=int(mask.sum());all_cells+=mask.numel()
            for i,meta in enumerate(batch.metadata):
                m=mask[i];matrix=torch.bincount(truth[i][m]*2+pred[i][m].long(),minlength=4).reshape(2,2).cpu().numpy()
                road_scenes.setdefault(meta['scene_id'],np.zeros((2,2),np.int64))[:]+=matrix
            jm=mask&targets['junction_support']
            junction_confusion+=torch.bincount(truth[jm]*2+pred[jm].long(),minlength=4).reshape(2,2).cpu().numpy()
        if first is None:
            first={'images':inputs['images'][0].cpu(),'depth_prediction':prediction[0].cpu(),
                   'depth_gt':targets['depth_m'][0].cpu(),'valid':targets['depth_valid'][0].cpu(),
                   'surface_prediction':output['surface_log_counts'][0].float().cpu(),
                   'surface_target':torch.log1p(counts[0]).cpu()}
            if 'drivable_logits' in output:
                first.update(drivable_prediction=output['drivable_logits'][0].float().sigmoid().cpu(),
                             drivable_target=targets['drivable_static'][0].cpu(),drivable_valid=targets['drivable_valid'][0].cpu())
    drivable={'confusion_tn_fp_fn_tp':confusion.tolist(),'valid_cells':valid_cells,'total_cells':all_cells,
        'iou_positive':float(confusion[1,1]/max(1,confusion[1,1]+confusion[0,1]+confusion[1,0])) if valid_cells else None,
        'valid_fraction':valid_cells/all_cells if all_cells else None,
        'scope':'static lane support on declared valid mask; optional junction ribbons; not traffic legality',
        'per_scene':{k:road_metrics(v) for k,v in road_scenes.items()},'junction':road_metrics(junction_confusion)}
    return {'drivable_static':drivable,'samples':len(indices),'sample_ids':sample_ids,'loss':loss_average(totals,len(indices)),
            'depth':metrics.result(),'depth_per_scene':{k:v.result() for k,v in scene_metrics.items()},'oracle_depth_bin_quantization':quantized.result(),
            'surface_log_count_mae':surface_sum/surface_cells},first


def write_preview(example,root,name):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    panels=[]
    for cam in range(6):
        rgb=(example['images'][cam].numpy().transpose(1,2,0)*255).round().astype(np.uint8)[...,::-1].copy()
        maps=[]
        for key in ('depth_gt','depth_prediction'):
            depth=example[key][cam].numpy();valid=example['valid'][cam].numpy()
            colors=cv2.applyColorMap((np.clip(depth/80,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
            colors[~valid]=0
            maps.append(cv2.resize(colors,(320,192),interpolation=cv2.INTER_NEAREST))
        rgb=cv2.resize(rgb,(320,192))
        panel=np.concatenate((rgb,*maps),axis=1)
        panel=cv2.copyMakeBorder(panel,26,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(panel,'RGB | GT optical Z | RGB predicted Z (same GT evaluation mask)',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        panels.append(panel)
    cv2.imwrite(str(root/f'{name}_depth.png'),np.concatenate(panels,axis=0))
    # X forward up, Y left left: flip both spatial axes only for display.
    bev=[]
    maximum=max(float(example['surface_target'].max()),1.)
    for key in ('surface_target','surface_prediction'):
        layers=[]
        for layer in example[key].numpy():
            pixels=(np.clip(layer[::-1,::-1]/maximum,0,1)*255).astype(np.uint8)
            layers.append(cv2.resize(cv2.applyColorMap(pixels,cv2.COLORMAP_TURBO),(240,320),interpolation=cv2.INTER_NEAREST))
        panel=np.concatenate(layers,1)
        panel=cv2.copyMakeBorder(panel,30,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(panel,key+' | four height layers | log(1+sampled visible surfaces)',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        bev.append(panel)
    cv2.imwrite(str(root/f'{name}_bev.png'),np.concatenate(bev,axis=0))
    if 'drivable_prediction' in example:
        masks=[];valid=example['drivable_valid'].numpy()
        for key in ('drivable_target','drivable_prediction'):
            value=example[key].numpy();color=np.full((*value.shape,3),60,np.uint8)
            color[valid & (value>=.5)]=[50,190,50];color[valid & (value<.5)]=[40,40,220]
            masks.append(cv2.resize(color[::-1,::-1],(480,640),interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(root/f'{name}_drivable.png'),np.concatenate(masks,axis=1))



def selection_score(report,config):
    metric=config.get('selection_metric','total_loss')
    if metric=='depth_mae':return report['depth']['all']['mae_m']
    if metric=='depth_macro_abs_rel':
        bands=('1.5-10m','10-30m','30-60m','60-80m')
        values=[report['depth'][b]['abs_rel'] for b in bands]
        if any(v is None for v in values):raise ValueError('Selection requires all four depth ranges')
        return sum(values)/len(values)
    if metric=='total_loss':return report['loss']['total']
    raise ValueError('Unknown checkpoint selection metric')


def road_metrics(matrix):
    tn,fp,fn,tp=matrix.ravel().tolist();n=tn+fp+fn+tp
    return {'confusion_tn_fp_fn_tp':matrix.tolist(),'valid_cells':n,
            'iou_positive':tp/(tp+fp+fn) if tp+fp+fn else None,
            'recall_positive':tp/(tp+fn) if tp+fn else None,
            'all_positive_iou':(tp+fn)/n if n else None}
