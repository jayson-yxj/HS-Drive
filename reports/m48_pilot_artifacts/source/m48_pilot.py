"""Fixed two-frame CUDA overfit and per-cell prior; never a generalization score."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from datasets.m48_training import EvidenceDataset
from datasets.inputs import INPUT_FIELDS
from models.m48_evidence_head import EvidenceHead,PositionPrior
from training.m48_evidence_loss import evidence_loss,evidence_targets
from training.runner import initialize,load_checkpoint
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import freeze_sources


def scores(pred,labels):
    result={}
    for key,(target,mask) in evidence_targets(labels).items():
        binary=pred[key]>0;positive=target.bool();tp=(binary&positive&mask).sum().item();fp=(binary&~positive&mask).sum().item();fn=(~binary&positive&mask).sum().item();tn=(~binary&~positive&mask).sum().item()
        result[key]=dict(tp=tp,fp=fp,fn=fn,tn=tn,iou=tp/max(1,tp+fp+fn),positive_recall=tp/max(1,tp+fn),negative_recall=tn/max(1,tn+fp),pred_positive_fraction=float(binary[mask].float().mean()),target_positive_fraction=float(positive[mask].float().mean()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m48_occupancy_pilot.json');p.add_argument('--run',default='runs/m48_pilot');p.add_argument('--steps',type=int,default=400);a=p.parse_args()
    if Path(a.output).exists() or Path(a.run).exists():raise FileExistsError('New run required')
    root=Path(a.run);root.mkdir();art=Path('reports/m48_pilot_artifacts');art.mkdir()
    prepare_cuda();torch.manual_seed(48);torch.set_num_threads(2)
    sources=freeze_sources(art/'source',[__file__,'datasets/m48_training.py','models/m48_evidence_head.py','training/m48_evidence_loss.py'])
    ds=EvidenceDataset('/root/gpufree-data/hs-drive-data/processed/m46_training_dense_index','train','reports/m48_spatial_evidence.json')
    ids=[next(i for i,r in enumerate(ds.rows) if r['sample_id'].split('/')[0]==scene) for scene in ['road_a','near_a']]
    items=[ds[i] for i in ids];assert all(set(x.inputs)==INPUT_FIELDS for x in items)
    cfg=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text())['config']
    encoder=initialize(cfg,pretrained=False).cuda().eval();load_checkpoint(Path('runs/m4_expanded_joint/best.pt'),encoder)
    for p in encoder.parameters():p.requires_grad_(False)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();features=[];geometry_dtypes=[]
    def hook(module,args,output):geometry_dtypes.append([str(x.dtype) for x in output])
    handle=encoder.splat.register_forward_hook(hook)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for item in items:features.append(encoder(**{k:v[None].cuda() for k,v in item.inputs.items()})['bev_current'].float())
    handle.remove();features=torch.cat(features);labels={k:torch.stack([i.targets[k] for i in items]).cuda() for k in items[0].targets}
    # Reliability subset determined by label evidence only; no heldout/val selection.
    reliable=~labels['evidence_conflict_mask'] & ~labels['semantic_conflict_mask'] & ~labels['ego_mask']
    for k in ['surface_valid','free_evidence_valid','surface_semantic_valid','obstacle_valid']:labels[k]&=reliable
    assert all(x==['torch.float32','torch.float32'] for x in geometry_dtypes)
    assert not any(p.grad is not None for p in encoder.parameters())
    records=[]
    for name,model,lr in [('rgb_bev',EvidenceHead().cuda(),.003),('position_prior',PositionPrior().cuda(),.05)]:
        optimizer=torch.optim.Adam(model.parameters(),lr=lr);history=[];started=time.perf_counter();finite=True
        for step in range(a.steps+1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(features)
            loss,parts=evidence_loss(out,labels)
            if step%20==0 or step==a.steps:
                history.append(dict(step=step,total=float(loss.detach()),tasks={k:float(v.detach()) for k,v in parts.items()}))
            if step==a.steps:break
            loss.backward();grad=max(float(p.grad.detach().abs().max()) for p in model.parameters() if p.grad is not None)
            finite &= bool(np.isfinite(grad));optimizer.step()
        torch.cuda.synchronize();metrics=scores(out,labels)
        # Unknown gradient and forbidden fields tested on the real training labels.
        probe={k:v.detach().float().requires_grad_() for k,v in out.items()};pl,_=evidence_loss(probe,labels);pl.backward()
        ignored={k:not bool(probe[k].grad[~mask].any()) for k,(_,mask) in evidence_targets(labels).items()}
        ck=root/(name+'.pt');torch.save(model.state_dict(),ck)
        np.savez_compressed(art/(name+'_predictions.npz'),**{k:torch.sigmoid(v.float()).detach().cpu().numpy() for k,v in out.items()})
        records.append(dict(name=name,steps=a.steps,lr=lr,history=history,metrics=metrics,finite_gradients=finite,last_gradient_max=grad,ignored_gradient_zero=ignored,elapsed_s=time.perf_counter()-started,checkpoint=str(ck),sha256=file_hash(ck),parameters=sum(p.numel() for p in model.parameters())))
        print(name,metrics,flush=True)
    np.savez_compressed(art/'labels.npz',**{k:v.cpu().numpy() for k,v in labels.items()})
    atomic_json(a.output,dict(kind='actual_RTX4090_two_frame_overfit_no_generalization',records=records,samples=[x.metadata for x in items],
        geometry_dtypes=geometry_dtypes,backbone_fully_frozen=True,depth_bins_changed=False,splat_changed=False,
        checkpoint_sha256=file_hash('runs/m4_expanded_joint/best.pt'),sources=sources,seed=48,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_s=time.perf_counter()-start,
        formal_training_approved=False,scope='Event-mask learnability only. Position prior can memorize fixed scene geometry. No checkpoint selection on validation.'))
if __name__=='__main__':main()
