"""Read-only actual tensor instrumentation of frozen official PETR."""
import sys,json,inspect,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from tools.m8c_common import CFG,ROOT,BASELINE,RUNTIME,asset,write,provenance,stats

def main():
    out=ROOT/'hook';out.mkdir(exist_ok=False)
    # Deny evaluator/raw references in Python IO. No network namespace in launcher.
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if str(p).startswith('/root/gpufree-data/hs-drive-data') or Path.cwd()/'runs' in p.parents or p.name in ['evaluation_index.json','evaluation_ledger.json']:
                raise PermissionError('Evaluator paths not accessible to inference')
    sys.addaudithook(guard)
    torch.manual_seed(8082);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from datasets.m8b_petr_adapter import PETRCarlaOfflineAdapter,old_boxes_to_hs
    cfg=mmcv.Config.fromfile(str(RUNTIME/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'));cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    safe=RUNTIME/'artifacts/safe_extract/model_tensors.npz';assert sha_file(safe)==CFG['numeric_weights_sha256']
    with np.load(safe,allow_pickle=False) as arrays:state={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False);head=model.pts_bbox_head
    adapter=PETRCarlaOfflineAdapter(BASELINE/'rgb_bundle');method=head.position_embeding.__func__;lines,start=inspect.getsourcelines(method)
    capture_line=start+next(i for i,l in enumerate(lines) if 'coords_mask = (coords3d >' in l)
    observed={}
    def tracer(frame,event,arg):
        if frame.f_code is method.__code__:
            if event=='line' and frame.f_lineno==capture_line:
                observed['normalized_xyz']=frame.f_locals['coords3d'].detach().cpu().numpy().copy()
                observed['depth_m']=frame.f_locals['coords_d'].detach().cpu().numpy().copy()
                observed['img2lidars']=frame.f_locals['img2lidars'][0,:,0,0,0].detach().cpu().numpy().copy()
            if event=='return':observed['geometric_mask']=arg[1].detach().cpu().numpy().copy()
            return tracer
        return None
    def prehook(module,args):observed['position_encoder_logits']=args[0].detach().cpu().numpy().copy()
    timings=[];records=[]
    for token in CFG['hook_tokens']:
        packet_path=BASELINE/'rgb_bundle'/token/'packet.json';p=json.loads(packet_path.read_text());cpu,meta,cal,times=adapter.prepare(p);inp=torch.from_numpy(cpu).cuda()
        def forward(hook):
            observed.clear();handle=head.position_encoder.register_forward_pre_hook(prehook) if hook else None
            torch.cuda.synchronize();t=time.perf_counter()
            if hook:sys.settrace(tracer)
            try:
                with torch.no_grad():
                    feats=model.extract_feat(inp.view(1,6,3,512,1408),[dict(meta)]);raw=head(feats,[dict(meta)]);result=head.get_bboxes(raw,[dict(meta)],rescale=True)[0]
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            finally:
                sys.settrace(None)
                if handle:handle.remove()
            return raw,result,elapsed
        if token==CFG['hook_tokens'][0]:
            forward(False)
            for _ in range(3):
                a,aa,ta=forward(False);b,bb,tb=forward(True)
                assert torch.equal(a['all_bbox_preds'],b['all_bbox_preds']) and torch.equal(a['all_cls_scores'],b['all_cls_scores'])
                assert torch.equal(aa[0].tensor,bb[0].tensor);timings.append(dict(no_hook_ms=ta,hook_ms=tb,overhead_ms=tb-ta))
        raw,result,ms=forward(True)
        norm=observed['normalized_xyz'][0];mask=observed['geometric_mask'][0];D=norm.shape[-2]
        np.savez_compressed(out/(token+'_tensors.npz'),**observed,all_bbox_preds=raw['all_bbox_preds'].detach().cpu().numpy(),all_cls_scores=raw['all_cls_scores'].detach().cpu().numpy(),learned_query_reference=head.reference_points.weight.detach().cpu().numpy())
        camera=[]
        for ci in range(6):
            x=norm[ci];outside=(x<0)|(x>1)
            camera.append(dict(camera=p['cameras'][ci]['name'],normalized_axes=[stats(x[...,j]) for j in range(3)],outside_any_xyz_fraction=float(outside.any(-1).mean()),outside_axis_fraction=outside.mean((0,1,2)).tolist(),computed_geometric_mask_fraction=float(mask[ci].mean()),outside_by_depth=outside.any(-1).mean((0,1)).tolist(),outside_by_depth_axis=outside.mean((0,1)).tolist(),feature_cells=int(mask[ci].size)))
        box=result[0].tensor.cpu().numpy();scores=result[1].cpu().numpy();labels=result[2].cpu().numpy();centers=raw['all_bbox_preds'][:,:,:, [0,1,4]].detach().cpu().numpy()
        rec=dict(token=token,frame_id=p['frame_id'],timestamp_s=p['timestamp_s'],packet=asset(packet_path),calibration=cal,kind='ActualRTX4090TensorHook',input_shape=list(cpu.shape),feature_grid=list(norm.shape),depth_samples=observed['depth_m'].tolist(),position_range=list(head.position_range),pc_range=list(head.pc_range),camera=camera,query_center_axes_per_layer=[[stats(centers[layer,...,axis]) for axis in range(3)] for layer in range(6)],forward_with_hook_ms=ms,raw_boxes=box.tolist(),boxes=old_boxes_to_hs(box).tolist(),scores=scores.tolist(),labels=labels.tolist(),all_query_tensors=asset(out/(token+'_tensors.npz')))
        write(out/(token+'.json'),rec);records.append(rec);print(token,ms,'mask',[round(c['computed_geometric_mask_fraction'],3) for c in camera],flush=True)
    model.cpu();assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    write(out/'run.json',dict(**provenance('ActualRTX4090FrozenTensorObservation'),device=torch.cuda.get_device_name(),records=records,hook_overhead_paired=timings,hook_on_off_bitwise_equal=True,weights_unchanged=True,trace_line=capture_line,official_source=asset(inspect.getfile(method)),limitation='Computed position geometry mask is returned then discarded by official forward; original pad mask used by transformer. Outside0to1 values clamped by inverse_sigmoid. Copy/sync instrumentation affects runtime, not production timing.'))
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':main()
