"""Frozen PETR inference process: only sanitized six-RGB numeric bundle."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np


def main():
    inp=Path(os.environ['M8J4_INPUT']);out=Path(os.environ['M8J4_OUTPUT'])
    runtime=Path(os.environ['M8J4_RUNTIME'])
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            if ('/sidecar/' in p or '/extracted/' in p or '/validation/' in p
                    or '/reports/' in p or '/hs-drive-data/' in p
                    or ('/hs-drive-private/' in p and not (p.startswith(str(inp)+'/') or p.startswith(str(out)+'/')))):
                raise PermissionError('GT/supervision forbidden in model process')
    sys.addaudithook(guard)
    canary=False
    try:Path('/blocked/sidecar/00000.json').read_bytes()
    except PermissionError:canary=True
    assert canary
    import torch
    torch.set_num_threads(2);torch.manual_seed(8082)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from datasets.m8j4_petr_input import prepare
    from datasets.m8b_petr_adapter import old_boxes_to_hs,corners_hs
    manifest=json.loads((inp/'manifest.json').read_text())
    config=mmcv.Config.fromfile(str(runtime/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    config.model.img_backbone.pretrained=None
    model=build_model(config.model,train_cfg=config.get('train_cfg'),test_cfg=config.get('test_cfg'))
    weights=runtime/'artifacts/safe_extract/model_tensors.npz'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==manifest['model']['weights_sha256']
    with np.load(weights,allow_pickle=False) as arrays:state={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False)
    head=model.pts_bbox_head;assert len(state)==558 and head.num_query==900 and head.num_classes==10
    records=[]
    for entry in manifest['entries']:
        path=inp/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files)=={'images','intrinsics','cam_to_ego'}
            rgb,k,t=arrays['images'],arrays['intrinsics'],arrays['cam_to_ego']
        cpu,meta,audit=prepare(rgb,k,t);preprocess=1000*(time.perf_counter()-start)
        tick=time.perf_counter();tensor=torch.from_numpy(cpu).cuda();torch.cuda.synchronize();h2d=1000*(time.perf_counter()-tick)
        with torch.no_grad():
            tick=time.perf_counter();features=model.extract_feat(tensor,[dict(meta)])
            raw=head(features,[dict(meta)]);torch.cuda.synchronize();forward=1000*(time.perf_counter()-tick)
            assert all(torch.isfinite(raw[x]).all().item() for x in ('all_cls_scores','all_bbox_preds'))
            tick=time.perf_counter();decoded=head.get_bboxes(raw,[dict(meta)],rescale=True)[0]
            torch.cuda.synchronize();decode=1000*(time.perf_counter()-tick)
            boxes=decoded[0].tensor.cpu().numpy();scores=decoded[1].cpu().numpy();labels=decoded[2].cpu().numpy()
        hs=old_boxes_to_hs(boxes)
        official=LiDARInstance3DBoxes(torch.from_numpy(boxes),box_dim=9).corners.numpy()
        corner_error=max(float(np.min(np.linalg.norm(o[:,None]-corners_hs(h)[None],axis=-1),axis=1).max()) for o,h in zip(official,hs))
        assert boxes.shape==(300,9) and np.isfinite(hs).all() and np.isfinite(scores).all() and corner_error<1e-4
        rec=dict(ordinal=entry['ordinal'],source_frame=None,timestamp_s=None,temporal_valid=False,
            source='ExistingFrozenPETR_RGBOnly_Offline',control_authorized=False,
            coordinate='HS_ego_x_forward_y_left_z_up',box_fields=['cx','cy','cz','length','width','height','yaw_ccw','vx_unvalidated','vy_unvalidated'],
            label_names=list(config.class_names),calibration=audit,
            shapes=dict(input=list(tensor.shape),features=[list(x.shape) for x in features],
                all_cls_scores=list(raw['all_cls_scores'].shape),all_bbox_preds=list(raw['all_bbox_preds'].shape),decoded=list(boxes.shape)),
            boxes_hs=hs.tolist(),scores=scores.tolist(),labels=labels.tolist(),
            corner_contract_max_m=corner_error,finite=True,
            latency_ms=dict(read_preprocess=preprocess,h2d=h2d,model_forward=forward,decode=decode,
                total_before_write=1000*(time.perf_counter()-start)),
            cold_first_call=len(records)==0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (out/(entry['ordinal']+'.json')).open('x') as f:json.dump(rec,f,allow_nan=False)
        records.append({k:v for k,v in rec.items() if k not in ('boxes_hs','scores','labels','calibration')})
        print(entry['ordinal'],rec['latency_ms'],rec['peak_allocated_bytes'],flush=True)
        del rgb,k,t,cpu,tensor,features,raw,decoded
    after=model.cpu().state_dict();assert all(torch.equal(after[k],v) for k,v in state.items())
    with (out/'run.json').open('x') as f:json.dump(dict(status='FrozenRealSixRGBForwardGo',records=records,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        strict_tensors=len(state),parameters=sum(p.numel() for p in model.parameters()),
        weights_unchanged=True,training_steps=0,GT_canary_rejected=canary,
        input_manifest_sha256=hashlib.sha256((inp/'manifest.json').read_bytes()).hexdigest(),
        weights_sha256=manifest['model']['weights_sha256'],warmup_calls=0,forward_calls=3,
        quality='NotCertified_NoAPOrGeneralizationClaim',network='DisabledByLauncher'),f,indent=2)


if __name__=='__main__':main()
