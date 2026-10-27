"""Two fixed checkpoints, read-only RGB inference and query provenance."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

from tools import m8j6_gpu_actual as old
from tools.m8j7_gpu import visualize
from training.m8j6_retry_state import snapshot,fingerprint,assert_exact
from datasets.m8j6_selection import sha256


def query_provenance(head,raw,decoded,scores,labels):
    coder=head.bbox_coder
    s,index=raw['all_cls_scores'][-1,0].sigmoid().flatten().topk(coder.max_num)
    queries=index//head.num_classes;classes=index%head.num_classes
    selected=raw['all_bbox_preds'][-1,0,queries]
    low=selected.new_tensor(coder.post_center_range[:3]);high=selected.new_tensor(coder.post_center_range[3:])
    center=selected[:,[0,1,4]]
    keep=((center>=low)&(center<=high)).all(1)
    assert torch.equal(s[keep],scores) and torch.equal(classes[keep],labels)
    assert torch.allclose(selected[keep,4],decoded.gravity_center[:,2],atol=1e-6,rtol=1e-5)
    assert torch.allclose(selected[keep,5].exp(),decoded.tensor[:,5],atol=1e-6,rtol=1e-5)
    return queries[keep].cpu().numpy(),index[keep].cpu().numpy()


def main():
    protocol=json.loads(Path('configs/m8j8_protocol.json').read_text());cfg=json.loads(Path('configs/m8j6_protocol.json').read_text())
    phase=os.environ['M8J8_PHASE'];rt=Path(os.environ['M8J6_RUNTIME']);rep=Path(os.environ['M8J8_REPORT'])
    root=Path(protocol['prepared_root']).parent;private=root/phase;private.mkdir(exist_ok=False)
    prepared=Path(protocol['prepared_root']);manifest=json.loads((prepared/'manifest.json').read_text())
    oldpre=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    oldmanifest=json.loads((oldpre/'manifest.json').read_text())
    result=dict(model=phase,code_commit=os.environ['M8J8_COMMIT'],started_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Running',optimizer_steps=0,backward_calls=0,precision='FP32',AMP=False,TF32=False,
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,
        mmcv=old.mmcv.__version__,mmdet=old.mmdet.__version__,mmdet3d=old.mmdet3d.__version__,frames=[])
    start=time.perf_counter()
    try:
        seed=protocol['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(False)
        assert sha256(protocol['A']['path'])==protocol['A']['sha256']
        model=old.build(rt,cfg,Path(protocol['A']['path']))
        if phase=='B':
            cp=protocol['B'];assert sha256(cp['path'])==cp['sha256']
            payload=torch.load(cp['path'],map_location='cpu') # own hashed checkpoint in isolated process
            assert payload['global_step']==400 and fingerprint(payload['model'])==cp['model_fingerprint']
            model.load_state_dict(payload['model'],strict=True)
            assert_exact(payload['model'],model.state_dict(),'BCheckpointLoad')
            del payload
        model.eval();initial=snapshot(model.state_dict());initial_hash=fingerprint(initial)
        assert all(not m.training for m in model.modules())
        assert model.pts_bbox_head.bbox_coder.max_num==300
        result['initial_state_fingerprint']=initial_hash
        # Original requires_grad flags / alias relations are unchanged; no optimizer is constructed.
        aliases={}
        for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
        alias_groups=list(aliases.values())
        sys.addaudithook(old.input_guard)
        old.PERCEPTION_ACTIVE=True;rejected=0
        for path in [prepared/'supervision/00.npz',oldpre.parent/'data/samples/LIDAR_TOP/forbidden.bin']:
            try:path.read_bytes()
            except PermissionError:rejected+=1
        old.PERCEPTION_ACTIVE=False;assert rejected==2
        result['input_canaries_rejected']=rejected
        jobs=[('residual',oldpre,oldmanifest,oldmanifest['frames']),('mini_val',prepared,manifest,manifest['frames'])]
        for kind,prep,mf,frames in jobs:
            output=private/kind;output.mkdir(exist_ok=False)
            visual_ids=set(range(8)) if kind=='residual' else {0,20,39,40,60,80}
            for frame in frames:
                t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    img,meta,gt,gt_labels=old.load_frame(prep,frame)
                    raw=old.forward(model,img,meta)
                    decoded,scores,labels=model.pts_bbox_head.get_bboxes(raw,meta)[0]
                    old.finite(decoded.tensor,'decoded');old.finite(scores,'scores')
                    assert (decoded.tensor[:,3:6]>0).all() and len(gt)==frame['retained_gt']
                    qi,flat=query_provenance(model.pts_bbox_head,raw,decoded,scores,labels)
                    record=dict(boxes=decoded.tensor.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),
                        gt=gt.tensor.cpu().numpy(),gt_labels=gt_labels.cpu().numpy(),query_indices=qi,flat_indices=flat,
                        raw_cls=raw['all_cls_scores'][-1,0].cpu().numpy(),raw_bbox=raw['all_bbox_preds'][-1,0].cpu().numpy())
                    if kind=='residual':
                        cache=Path('/root/gpufree-data/hs-drive-private/m8j7_20261027_132740')/('eval_000' if phase=='A' else 'eval_400')/f'{frame["index"]:02}_decoded_and_GT.npz'
                        with np.load(cache,allow_pickle=False) as z:
                            for key in z.files:
                                np.testing.assert_allclose(record[key],z[key],atol=1e-6,rtol=1e-5)
                    path=output/f'{frame["index"]:02}.npz';np.savez_compressed(path,**record)
                    if frame['index'] in visual_ids:
                        visualize(output,frame['index'],img,meta,gt,decoded,record['scores'],.2,mf['six_camera_order'],0 if phase=='A' else 400)
                assert_exact(initial,model.state_dict(),'ReadOnlyFrame')
                assert all(p.grad is None for p in model.parameters())
                assert all(not m.training for m in model.modules())
                torch.cuda.synchronize()
                result['frames'].append(dict(dataset=kind,index=frame['index'],sample_token=frame['sample_token'],
                    scene_name=frame['scene_name'],GT=len(gt),decoded=len(scores),cache_sha256=sha256(path),
                    cache_path=str(path),seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    parameters_buffers_exact_unchanged=True,old_J7_cache_matches=True if kind=='residual' else None))
                print(phase,kind,frame['index'],'GT',len(gt),'decoded',len(scores),flush=True)
                del img,meta,gt,gt_labels,raw,decoded,scores,labels,record
        current={}
        for name,p in model.named_parameters(remove_duplicate=False):current.setdefault(id(p),[]).append(name)
        assert list(current.values())==alias_groups
        assert_exact(initial,model.state_dict(),'FinalReadOnly')
        result.update(status='ReadOnlyEvaluationPassed',final_state_fingerprint=fingerprint(model.state_dict()),
            parameters_buffers_unchanged=True,alias_relationships_unchanged=True,all_modules_eval=True,
            requires_grad_flags_changed=False,old_eight_frame_caches_match=True)
    except Exception as exc:
        result.update(status='FailedStopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(elapsed_s=time.perf_counter()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
        old.save(rep/'result.json',result)


if __name__=='__main__':main()
