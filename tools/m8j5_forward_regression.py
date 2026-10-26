"""One frozen RGB-only regression forward; zero real backward/update calls."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    root=Path(os.environ['M8J5_PRIVATE']);out=root/os.environ['M8J5_FORWARD_SUBDIR'];out.mkdir(exist_ok=False)
    inp=Path('/root/gpufree-data/hs-drive-private/m8j4-single-frame/inference_input')
    runtime=Path(os.environ['M8J5_RUNTIME'])
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            if ('/sidecar/' in p or '/extracted/' in p or '/validation/' in p or '/reports/' in p
                or ('/hs-drive-private/' in p and not (p.startswith(str(inp)+'/') or p.startswith(str(out)+'/')))):
                raise PermissionError('GT/review data forbidden during inference')
    sys.addaudithook(guard)
    try:Path('/blocked/sidecar/canary.json').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('GT guard failed')
    import torch
    torch.set_num_threads(2);torch.manual_seed(8082)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from datasets.m8j4_petr_input import prepare
    from training.m8j5_ignore_loss import IgnoreAwarePETRLoss,Supervision
    cfg=mmcv.Config.fromfile(str(runtime/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    weight=runtime/'artifacts/safe_extract/model_tensors.npz'
    assert hashlib.sha256(weight.read_bytes()).hexdigest()=='cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591'
    with np.load(weight,allow_pickle=False) as z:state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False)
    adapter=IgnoreAwarePETRLoss(model.pts_bbox_head)
    manifest=json.loads((inp/'manifest.json').read_text());entry=next(e for e in manifest['entries'] if e['ordinal']=='00107')
    path=inp/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'images','intrinsics','cam_to_ego'}
        cpu,meta,audit=prepare(z['images'],z['intrinsics'],z['cam_to_ego'])
    x=torch.from_numpy(cpu).cuda();input_shape=list(x.shape)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad():
        features=model.extract_feat(x,[dict(meta)]);raw=model.pts_bbox_head(features,[dict(meta)])
        decoded=model.pts_bbox_head.get_bboxes(raw,[dict(meta)],rescale=True)[0]
        torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
        # No GT is read. Current unsigned supervision authorizes zero targets/regions.
        spec=Supervision(torch.empty((0,9),device='cuda'),torch.empty(0,device='cuda',dtype=torch.long),
            torch.empty(0,device='cuda',dtype=torch.bool),torch.empty((0,6),device='cuda'),torch.empty((0,6),device='cuda'),{})
        losses,counts=adapter(raw,[spec])
    assert all(torch.isfinite(v).all() for v in raw.values() if isinstance(v,torch.Tensor))
    assert all(float(v)==0 for v in losses.values())
    np.savez(out/'predictions.npz',boxes=decoded[0].tensor.cpu().numpy(),scores=decoded[1].cpu().numpy(),labels=decoded[2].cpu().numpy())
    result=dict(status='FrozenForwardGo_ZeroRealUpdates',ordinal='00107',input_shape=input_shape,
        raw_class_shape=list(raw['all_cls_scores'].shape),raw_box_shape=list(raw['all_bbox_preds'].shape),
        finite=True,losses={k:float(v) for k,v in losses.items()},query_counts=counts,
        forward_and_decode_ms=latency,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name(),
        supervision='NoGoldenNoCertifiedNegative_900IgnorePerLayer',GT_canary_rejected=True,
        real_backward_calls=0,optimizer_steps=0,weights_unchanged=True,source_frame=None,timestamp_s=None)
    after=model.cpu().state_dict();assert all(torch.equal(after[k],v) for k,v in state.items())
    with (out/'run.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='query_counts'}))


if __name__=='__main__':main()
