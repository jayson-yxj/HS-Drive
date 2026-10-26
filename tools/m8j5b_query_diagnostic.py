"""Four frozen forwards + loss-only audit; zero model backward/optimizer steps."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np


def main():
    root=Path(os.environ['M8J5B_PRIVATE']);out=Path(os.environ['M8J5B_DIAGNOSTIC_OUTPUT']);out.mkdir(exist_ok=False)
    inp=root/'inference_input';runtime=Path(os.environ['M8J5_RUNTIME']);phase={'forward':False}
    def guard(event,args):
        if phase['forward'] and event=='open' and isinstance(args[0],(str,bytes)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            if '/hs-drive-private/' in p and not (p.startswith(str(inp)+'/') or p.startswith(str(out)+'/')):
                raise PermissionError('Supervision forbidden during RGB forward')
    sys.addaudithook(guard)
    import torch
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from datasets.m8j4_petr_input import prepare
    from datasets.m8j5b_loss_adapter import load_coverage_supervision
    from training.m8j5b_coverage_loss import CoverageAwarePETRLoss
    torch.set_num_threads(2);torch.manual_seed(8082)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    cfg=mmcv.Config.fromfile(str(runtime/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    weights=runtime/'artifacts/safe_extract/model_tensors.npz'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()=='cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591'
    with np.load(weights,allow_pickle=False) as z:state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False)
    loss=CoverageAwarePETRLoss(model.pts_bbox_head);results=[]
    manifest=json.loads((inp/'manifest.json').read_text())
    for entry in manifest['entries']:
        ordinal=entry['ordinal'];p=inp/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
        phase['forward']=True
        try:(root/'sidecar/canary.json').read_bytes()
        except PermissionError:pass
        else:raise AssertionError('GT isolation guard failed')
        with np.load(p,allow_pickle=False) as z:
            assert set(z.files)=={'images','intrinsics','cam_to_ego'}
            cpu,meta,_=prepare(z['images'],z['intrinsics'],z['cam_to_ego'])
        x=torch.from_numpy(cpu).cuda();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        cached=root/'predictions_run1'/(ordinal+'_raw.npz')
        with torch.no_grad():
            if cached.exists() and cached.parent!=out:
                # Resume one already completed frozen forward after its loss
                # calibration assertion failed. Do not repeat model compute.
                phase['forward']=False
                with np.load(cached,allow_pickle=False) as z:
                    pred=dict(all_cls_scores=torch.from_numpy(z['cls'].copy()).cuda(),
                              all_bbox_preds=torch.from_numpy(z['box'].copy()).cuda())
                elapsed=None;features=None;fresh=False
            else:
                features=model.extract_feat(x,[dict(meta)]);pred=model.pts_bbox_head(features,[dict(meta)])
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000;fresh=True
            assert all(torch.isfinite(v).all() for v in pred.values() if isinstance(v,torch.Tensor))
            np.savez_compressed(out/(ordinal+'_raw.npz'),cls=pred['all_cls_scores'].cpu().numpy(),box=pred['all_bbox_preds'].cpu().numpy())
            phase['forward']=False
            # Only now open privileged supervision. Nothing is passed to forward.
            s,record=load_coverage_supervision(root,ordinal)
            actual=np.asarray(meta['lidar2img'])[:,:3,:]
            residual=float(abs(actual-s.image_projection.cpu().numpy()).max());assert residual<.001
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as alerts:
                warnings.simplefilter('always');losses,audit=loss(pred,[s])
            torch.cuda.synchronize();loss_ms=(time.perf_counter()-t)*1000
        peak=torch.cuda.max_memory_allocated();assert peak<8*1024**3
        row=dict(ordinal=ordinal,source_boxes=len(s.boxes),layers=audit,losses={k:float(v) for k,v in losses.items()},
            warnings=sorted({str(a.message) for a in alerts}),forward_ms=elapsed,loss_audit_ms=loss_ms,
            fresh_model_call=fresh,cached_prediction_sha256=None if fresh else hashlib.sha256(cached.read_bytes()).hexdigest(),
            peak_allocated_bytes=peak,calibration_max_abs_error=residual,GT_guard_rejected=True,
            real_backward_calls=0,optimizer_steps=0)
        results.append(row);print(ordinal,[(a['positive'],a['ignore'],a['negative']) for a in audit],flush=True)
        del x,features,pred,s,losses
    after=model.cpu().state_dict();assert all(torch.equal(after[k],v) for k,v in state.items())
    report=dict(evidence='ActualFrozenSixRGBForward_ActualSidecar_LossOnly_NoBackward',
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,rows=results,weights_unchanged=True,
        new_model_calls=sum(r['fresh_model_call'] for r in results),total_model_calls_including_failed_run=4,
        negative_certificates=0,training=False,original_frame_and_timestamp=None,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['training/m8j5b_coverage_loss.py','datasets/m8j5b_loss_adapter.py']})
    with (out/'run.json').open('x') as f:json.dump(report,f,indent=2)
    # Publish aggregate counts, never predictions or per-query labels.
    public=json.loads(json.dumps(report))
    for row in public['rows']:
        for layer in row['layers']:
            for sample in layer['samples']:sample.pop('state')
    with Path('reports/m8j5b_real_query_diagnostic.json').open('x') as f:json.dump(public,f,indent=2)


if __name__=='__main__':main()
