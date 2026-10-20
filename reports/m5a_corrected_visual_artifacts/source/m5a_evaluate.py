"""Held-out event metrics and RGB interventions; thresholds fit on val only."""
import argparse,json,copy
from dataclasses import replace
from pathlib import Path
import numpy as np,torch
from datasets.m5a_training import VisualEvidenceDataset
from training.m48_evidence_loss import evidence_targets
from models.m48_evidence_head import EvidenceHead,PositionPrior
from tools.m5a_learning import encoder_load,stack,task_arrays
from diagnostics.m5a_metrics import binary_metrics,temperature_fit
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import freeze_sources


def arrays(outputs,labels):
    return {k:tuple(x.detach().cpu().numpy() for x in v) for k,v in task_arrays(outputs,labels).items()}

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))

def infer(net,head,items):
    pred={k:[] for k in ['surface','free','semantic']}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for item in items:
            out=head(net(**stack([item],'inputs'))['bev_current'])
            for k,v in out.items():pred[k].append(v.float().cpu())
    return {k:torch.cat(v) for k,v in pred.items()}

def summarize(arr,rows,thresholds):
    x=-20+(np.arange(160)+.5)*.5;y=-30+(np.arange(120)+.5)*.5;dist=np.hypot(x[:,None],y[None,:]);corr=(x[:,None]>=0)&(x[:,None]<=30)&(abs(y[None,:])<2)
    out={}
    for key,(logits,t,m) in arr.items():
        p=sigmoid(logits);th=thresholds[key];entry=dict(overall=binary_metrics(p,t,m,th),by_scene={},by_distance={},by_layer={},mask_fraction=float(m.mean()))
        for group in sorted(set(r['group'] for r in rows)):
            ids=[i for i,r in enumerate(rows) if r['group']==group];entry['by_scene'][group]=binary_metrics(p[ids],t[ids],m[ids],th)
        for lo,hi in [(0,10),(10,20),(20,40),(40,80)]:entry['by_distance'][f'{lo}_{hi}']=binary_metrics(p,t,m&((dist>=lo)&(dist<hi))[None,None],th)
        for z in range(4):entry['by_layer'][str(z)]=binary_metrics(p[:,z],t[:,z],m[:,z],th)
        vals=[v['iou'] for v in entry['by_scene'].values() if v['iou'] is not None];entry['scene_iou_mean']=float(np.mean(vals)) if vals else None;entry['scene_iou_std']=float(np.std(vals)) if vals else None
        if key=='obstacle':entry['future_straight_corridor']=binary_metrics(p,t,m&corr[None,None],th)
        out[key]=entry
    return out

def intervene(items,rows,kind):
    changed=[]
    for i,item in enumerate(items):
        new=copy.copy(item);inp=dict(item.inputs);rgb=inp['images'].clone()
        if kind=='mean_rgb':rgb=rgb.mean((-2,-1),keepdim=True).expand_as(rgb).clone()
        elif kind=='wrong_scene':
            j=next(j for j,r in enumerate(rows) if r['group']!=rows[i]['group'] and r['configuration']==rows[i]['configuration']);rgb=items[j].inputs['images'].clone()
        elif kind=='same_pose_swap':
            j=next(j for j,r in enumerate(rows) if r['group']==rows[i]['group'] and r['configuration']==(rows[i]['configuration']+1)%9);rgb=items[j].inputs['images'].clone()
        elif kind=='obstacle_occlusion':
            frame=json.loads(Path(rows[i]['raw_frame']).read_text());root=Path(rows[i]['raw_frame']).parent.parent
            for c,name in enumerate(rows[i]['camera_names']):
                with np.load(root/frame['cameras'][name]['instance']) as z:
                    sem=z['semantic'];mask=torch.from_numpy(np.isin(sem,[12,13,14,15,16,18,19]))
                rgb[c,:,mask]=rgb[c].mean((-2,-1))[:,None]
        inp['images']=rgb;changed.append(replace(item,inputs=inp))
    return changed

def paired_local(arr,rows):
    result=[]
    for group in sorted(set(r['group'] for r in rows)):
        zero=next(i for i,r in enumerate(rows) if r['group']==group and r['configuration']==0)
        for i,r in enumerate(rows):
            if r['group']!=group or r['configuration']==0:continue
            rec=dict(group=group,configuration=r['configuration'],tasks={})
            for key in ['surface','obstacle']:
                logits,t,m=arr[key];p=sigmoid(logits);common=m[i]&m[zero];changed=(t[i]!=t[zero])&common;delta=p[i]-p[zero];direction=t[i].astype(float)-t[zero].astype(float)
                rec['tasks'][key]=dict(changed_cells=int(changed.sum()),mean_abs_prediction_change=float(abs(delta[changed]).mean()) if changed.any() else None,directional_accuracy=float((delta[changed]*direction[changed]>1e-4).mean()) if changed.any() else None,mean_signed_correct_change=float((delta[changed]*direction[changed]).mean()) if changed.any() else None,unchanged_mean_abs_change=float(abs(delta[common&~changed]).mean()) if (common&~changed).any() else None)
            result.append(rec)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default='/root/gpufree-data/hs-drive-data/processed/m5a_index');p.add_argument('--run',default='runs/m5a_learning');p.add_argument('--artifacts',default='reports/m5a_visual_artifacts');p.add_argument('--prefix',default='reports/m5a');a=p.parse_args();art=Path(a.artifacts)
    if art.exists():raise FileExistsError(art)
    art.mkdir();sources=freeze_sources(art/'source',[__file__,'diagnostics/m5a_metrics.py','datasets/m5a_training.py']);prepare_cuda();torch.set_num_threads(2)
    ds={s:VisualEvidenceDataset(a.index,s) for s in ['val','test']};items={s:[d[i] for i in range(len(d))] for s,d in ds.items()};labs={s:stack(v,'targets','cpu') for s,v in items.items()}
    study=json.loads(Path('reports/m5a_training.json').read_text());records=[];diagnostics=[]
    for record in study['records']:
        name=record['name'];seed=record['seed'];net=encoder_load();head=(PositionPrior() if name=='A_prior' else EvidenceHead()).cuda().eval();ck=torch.load(record['checkpoint'],weights_only=True);head.load_state_dict(ck['head'])
        if 'context' in ck:net.context_head.load_state_dict(ck['context'])
        pred={s:infer(net,head,it) for s,it in items.items()};arr={s:arrays(pred[s],labs[s]) for s in items};thresholds={};calibration={}
        for key,(logits,t,m) in arr['val'].items():
            probs=sigmoid(logits);candidates={str(th):binary_metrics(probs,t,m,th)['iou'] for th in np.arange(1,10)/10};thresholds[key]=float(max(candidates,key=lambda k:candidates[k] if candidates[k] is not None else -1))
            cal=temperature_fit(logits,t,m);lp,lt,lm=arr['test'][key];cal['val_threshold_candidates']=candidates;cal['test_raw']=binary_metrics(sigmoid(lp),lt,lm,thresholds[key]);cal['test_calibrated_event_probability']=binary_metrics(sigmoid(lp/cal['temperature']+cal['bias']),lt,lm,.5);calibration[key]=cal
        summary=summarize(arr['test'],ds['test'].rows,thresholds);fixed=summarize(arr['test'],ds['test'].rows,{k:.5 for k in thresholds});local=paired_local(arr['test'],ds['test'].rows)
        saved=art/f'm5a_{name}_{seed}_test.npz';np.savez_compressed(saved,**{k:v.numpy() for k,v in pred['test'].items()})
        records.append(dict(name=name,seed=seed,test=summary,test_fixed_half=fixed,val_selected_thresholds=thresholds,calibration=calibration,local_pairs=local,predictions=str(saved),sha256=file_hash(saved)))
        for kind in ['mean_rgb','wrong_scene','obstacle_occlusion','same_pose_swap']:
            if name=='A_prior':perturbed=pred['test']
            else:perturbed=infer(net,head,intervene(items['test'],ds['test'].rows,kind))
            changed=arrays(perturbed,labs['test']);d={}
            for key,(l,t,m) in changed.items():
                original=sigmoid(arr['test'][key][0]);prob=sigmoid(l);d[key]=dict(scores=binary_metrics(prob,t,m,thresholds[key]),mean_abs_probability_change=float(abs(prob-original)[m].mean()),mean_abs_all_grid_change=float(abs(prob-original).mean()))
            diagnostics.append(dict(name=name,seed=seed,intervention=kind,tasks=d,diagnostic_only=True))
            print(name,seed,kind,flush=True)
        del net,head
        atomic_json(art/'m5a_progress.json',dict(records=records,diagnostics=diagnostics))
    comparisons=[]
    for r in records:
        if r['name']=='A_prior':continue
        prior=next(x for x in records if x['name']=='A_prior' and x['seed']==r['seed']);delta={}
        for key in ['surface','free','ground','obstacle']:
            delta[key]={g:r['test'][key]['by_scene'][g]['iou']-prior['test'][key]['by_scene'][g]['iou'] for g in r['test'][key]['by_scene']}
        comparisons.append(dict(name=r['name'],seed=r['seed'],scene_paired_iou_delta=delta))
    atomic_json(a.prefix+'_visual_generalization.json',dict(kind='actual_RTX4090_offline_inference_on_CARLA_new_pose_groups',records=records,sources=sources,test_samples=[r['sample_id'] for r in ds['test'].rows],test_groups=sorted(set(r['group'] for r in ds['test'].rows)),project_unseen_maps=False,selection='checkpoint and threshold/calibration val only',physical_probability=False,pr_auc_definition='noninterpolated precision-recall step integral (average precision), ties grouped',false_surface='FP under recipe event mask',missed_surface='FN under recipe event mask'))
    atomic_json(a.prefix+'_prior_comparison.json',dict(kind='paired_offline_test',comparisons=comparisons,capacity=study['records'],caveat='Three heldout pose groups only, correlated configurations; C warm-start adds compute, not equal-budget ablation.'))
    atomic_json(a.prefix+'_counterfactual_visual_tests.json',dict(kind='diagnostic_input_interventions_not_official_performance',records=diagnostics,local_pairs=[dict(name=r['name'],seed=r['seed'],pairs=r['local_pairs']) for r in records],mask_scope='semantic actor pixels used only to create diagnostic occlusion, never formal learned input'))
if __name__=='__main__':main()
