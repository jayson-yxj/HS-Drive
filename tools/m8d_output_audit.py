"""Actual tensor changes are separate from quality changes."""
import json
import numpy as np
from tools.m8d_common import *
from datasets.m8d_camera_order import INVERSE

def difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    d=np.abs(a.astype(float)-b.astype(float))
    return dict(shape=list(a.shape),exact_equal=bool(np.array_equal(a,b)),
                changed_fraction=float((a!=b).mean()),absolute=stats(d))

def main():
    rows=[]
    for token in CFG['hook_tokens']:
        with np.load(ROOT/'small'/(token+'_A_tensors.npz'),allow_pickle=False) as a, np.load(ROOT/'small'/(token+'_B_tensors.npz'),allow_pickle=False) as b:
            r=dict(token=token,raw_by_layer={k:[difference(a[k][i],b[k][i]) for i in range(6)] for k in ['all_cls_scores','all_bbox_preds']})
            r['position_by_physical_camera']={}
            for key in ['geometry_encoding','sine_encoding','adapted_sine_encoding','final_position_embedding']:
                x=a[key];y=b[key]
                if x.shape[0]==6:x=x[None];y=y[None]
                y=y[:,INVERSE]
                r['position_by_physical_camera'][key]=[difference(x[:,i],y[:,i]) for i in range(6)]
            r['query_center_delta_m_by_layer']=[stats(np.linalg.norm(
                a['all_bbox_preds'][i,0][:,[0,1,4]]-b['all_bbox_preds'][i,0][:,[0,1,4]],axis=-1)) for i in range(6)]
            r['query_center_distribution']={v:[[stats(z['all_bbox_preds'][i,0,:,axis]) for axis in [0,1,4]] for i in range(6)] for v,z in [('A',a),('B',b)]}
            try:
                with np.load(Path(CFG['m8c_root'])/'hook'/(token+'_tensors.npz'),allow_pickle=False) as c:
                    r['M8C_A_raw_replay']={k:difference(a[k],c[k]) for k in ['all_cls_scores','all_bbox_preds']}
            except FileNotFoundError:
                r['M8C_A_raw_replay']='NotAvailable'
        pa=json.loads((ROOT/'small'/(token+'_A.json')).read_text())
        pb=json.loads((ROOT/'small'/(token+'_B.json')).read_text())
        r['ranked_top300_difference']={k:difference(pa[k],pb[k]) for k in ['boxes','scores','labels']}
        r['top300_notice']='Rankwise differences are output differences, not object identity displacements.'
        rows.append(r)
    small=json.loads((ROOT/'small/run.json').read_text())
    write('reports/m8d_gpu_output_difference.json',dict(**provenance('ActualRTX4090ReadOnlyHooks'),
        rows=rows,A_replay=small['replay_A'],OrderEncodingChanged=True,
        OrderEffectOnPredictionsSupported=any(not d['exact_equal'] for r in rows for d in r['raw_by_layer']['all_cls_scores']),
        quality_inference='None; use paired evaluator. Geometry encoding aligned by physical camera; camera-index sine changes. All six layers and 900 queries saved externally.',
        artifacts=[asset(ROOT/'small'/(t+'_'+v+'_tensors.npz')) for t in CFG['hook_tokens'] for v in ['A','B']]))
    full=json.loads((ROOT/'full/run.json').read_text())
    timing={}
    for v in ['A','B']:
        allrows=[r for r in full['records'] if r['variant']==v]
        rr=[r for r in allrows if r['token']!='s000']
        timing[v]=dict(steady_frames=len(rr),timing_ms={k:stats([r['latency_ms'][k] for r in rr]) for k in rr[0]['latency_ms']},
            all192_timing_ms={k:stats([r['latency_ms'][k] for r in allrows]) for k in rr[0]['latency_ms']},
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in allrows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in allrows))
    write('reports/m8d_latency_resource.json',dict(**provenance('ActualRTX4090PairedFullRun_NoHooks'),
        variants=timing,hardware=full['device'],runtime=dict(torch=full['torch'],cuda=full['cuda']),
        pairing=CFG['full_pair_order'],full_A_replay=full['replay_A'],timing_contract=full['timing'],
        permutation='B adds explicit contiguous numpy camera-axis copy; A retains original adapter tensor. Copy time measured separately; same preprocessing and H2D.',
        preserved='M8B E2E P95 337.94ms and online NoGo remain. No 5Hz/20Hz/sharedGPU/CARLA certification.',
        source=asset(ROOT/'full/run.json')))
    contract=json.loads((ROOT/'contract/run.json').read_text())
    write('reports/m8d_permutation_contract.json',dict(**provenance('Actual192FrameRGBTensorGeometryContract'),
        status=contract['status'],frames=192,all_projection_residual_zero=all(c['pixel_projection_residual']==0 for r in contract['records'] for c in r['cameras']),
        image_SHA_multiset_equal=all(r['image_SHA_multiset_equal'] for r in contract['records']),
        inverse_exact=all(r['inverse_exact'] for r in contract['records']),
        all_camera_identity_records=contract['records'],source=asset(ROOT/'contract/run.json')))
if __name__=='__main__':main()
