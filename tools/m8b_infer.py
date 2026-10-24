"""Frozen official PETR inference. Network disabled by launcher, no evaluator imports."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    manifest_path=Path(args.manifest).resolve();bundle=manifest_path.parent;out=Path(args.output);out.mkdir(exist_ok=False)
    repo=Path.cwd();opened=set();blocked=[]
    def guard(event,arguments):
        if event!='open' or not isinstance(arguments[0],(str,bytes)):return
        p=Path(os.fsdecode(arguments[0])).resolve()
        forbidden=(str(p).startswith('/root/gpufree-data/hs-drive-data') or repo/'runs' in p.parents or repo/'reports' in p.parents or p.name=='evaluation_index.json')
        if forbidden:blocked.append(str(p));raise PermissionError('GT/evaluation path blocked')
        if bundle in p.parents:opened.add(str(p))
    sys.addaudithook(guard)
    for p in [repo/'runs/m5o_cache_train/supervision.json',bundle.parent/'evaluation_index.json']:
        try:p.read_bytes();raise AssertionError('Isolation canary failed')
        except PermissionError:pass
    import numpy as np
    import torch
    torch.manual_seed(8082);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from datasets.m8b_petr_adapter import PETRCarlaOfflineAdapter,old_boxes_to_hs,corners_hs
    m=json.loads(manifest_path.read_text());cfg=mmcv.Config.fromfile(m['petr_source']+'/projects/configs/petr/petr_r50dcn_gridmask_c5.py')
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    assert hashlib.sha256(Path(m['weights']['path']).read_bytes()).hexdigest()==m['weights']['sha256']
    with np.load(m['weights']['path'],allow_pickle=False) as arrays:state={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    model.load_state_dict(state,strict=True);assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert len(state)==558 and model.pts_bbox_head.num_query==900 and model.pts_bbox_head.num_classes==10 and len(model.pts_bbox_head.transformer.decoder.layers)==6
    model.cuda().eval();model.requires_grad_(False);adapter=PETRCarlaOfflineAdapter(bundle)
    # Independent official box corner agreement, including non-square boxes and velocity.
    boxes=torch.tensor([[12.,3.,.7,1.8,4.6,1.5,a,2.,-3.] for a in [-2.8,-1.57,0,.6,2.8]])
    official=LiDARInstance3DBoxes(boxes,box_dim=9).corners.numpy();converted=old_boxes_to_hs(boxes.numpy())
    corner_errors=[float(np.min(np.linalg.norm(o[:,None]-corners_hs(h)[None],axis=-1),axis=1).max()) for o,h in zip(official,converted)]
    assert max(corner_errors)<3e-6 and np.all(converted[:,7:]==[2,-3])
    records=[];contract=[]
    def save(name,obj):
        with (out/name).open('x') as f:json.dump(obj,f,allow_nan=False)
    for i,entry in enumerate(m['packets']):
        raw=Path(entry['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];p=json.loads(raw)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        cpu,meta,audit,times=adapter.prepare(p)
        # Projection identity in each actual camera, positive and negative optical depth.
        for c,proj in zip(p['cameras'],meta['lidar2img']):
            xyz=np.array([[-2,-1,5,1],[0,0,10,1],[2,2,20,1],[1,-1,-5,1]],float)
            ego=xyz@np.asarray(c['T_cam_to_ego']).T;projected=ego@proj.T
            assert np.all(projected[:3,2]>0) and projected[3,2]<0
            restored=projected@np.linalg.inv(proj).T
            contract.append(float(abs(restored-ego).max()))
        t=time.perf_counter();inp=torch.from_numpy(cpu).cuda();torch.cuda.synchronize();times['h2d_ms']=1000*(time.perf_counter()-t)
        with torch.no_grad():
            t=time.perf_counter();features=model.extract_feat(inp.view(1,6,3,512,1408),[dict(meta)]);head=model.pts_bbox_head(features,[dict(meta)]);torch.cuda.synchronize();times['model_forward_ms']=1000*(time.perf_counter()-t)
            t=time.perf_counter();obj=model.pts_bbox_head.get_bboxes(head,[dict(meta)],rescale=True)[0];torch.cuda.synchronize();times['box_decode_ms']=1000*(time.perf_counter()-t)
            t=time.perf_counter();box=obj[0].tensor.cpu().numpy();scores=obj[1].cpu().numpy();labels=obj[2].cpu().numpy();torch.cuda.synchronize();times['to_cpu_ms']=1000*(time.perf_counter()-t)
        assert np.isfinite(box).all() and np.isfinite(scores).all() and box.shape[1]==9
        hs=old_boxes_to_hs(box)
        record=dict(token=p['token'],schema='DetectedObjectHypothesis',source='OfficialPETR_Frozen_OfflineShadow',control_authorized=False,frame_id=p['frame_id'],timestamp_s=p['timestamp_s'],valid_until_sim_s=p['timestamp_s']+.35,
            completed_wall_utc=datetime.now(timezone.utc).isoformat(),coordinate_frame='HS_ego_x_forward_y_left_z_up',box_fields=['center_x','center_y','center_z','length','width','height','yaw_ccw','vx','vy'],
            unsupported=['physical_surface','safety_probability','free_space','control','validated_velocity','motion_compensation'],checkpoint_sha256=m['checkpoint_sha256'],packet_sha256=entry['sha256'],rgb_sha256=[c['sha256'] for c in p['cameras']],
            calibration=audit,raw_old_framework_boxes=box.tolist(),boxes=hs.tolist(),scores=scores.tolist(),labels=labels.tolist(),errors=[],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        t=time.perf_counter();payload=json.dumps(record,allow_nan=False,separators=(',',':'));times['serialization_ms']=1000*(time.perf_counter()-t);times['total_e2e_ms']=1000*(time.perf_counter()-start)
        record['latency_ms']=times;save(p['token']+'.json',record);records.append(dict(token=p['token'],latency_ms=times,count=len(scores),score_min=float(scores.min()),score_max=float(scores.max()),peak_allocated_bytes=record['peak_allocated_bytes'],peak_reserved_bytes=record['peak_reserved_bytes']))
        if i==0:
            # Staged profiling is required to equal the untouched official inference entry.
            with torch.no_grad():full=model(return_loss=False,rescale=True,img=[inp.view(1,6,3,512,1408)],img_metas=[[dict(meta)]])[0]['pts_bbox']
            assert np.array_equal(full['boxes_3d'].tensor.cpu().numpy(),box) and np.array_equal(full['scores_3d'].cpu().numpy(),scores)
        del inp,features,head,obj,cpu
        print(i,p['token'],'candidates',len(scores),'score_max',round(float(scores.max()),4),'total_ms',round(times['total_e2e_ms'],2),flush=True)
        if i==39:save('core40_completed.json',dict(frames=40,finite=True,calibration_contract=True,parameters_changed=False))
    after=model.cpu().state_dict();assert all(torch.equal(after[k],v) for k,v in state.items())
    save('run.json',dict(status='ActualRTX4090_RealCARLARGB_ForwardPassed',frames=len(records),records=records,training_steps=0,weights_unchanged=True,strict_tensors=558,parameters=sum(p.numel() for p in model.parameters()),
        device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,staged_official_forward_bitwise_equal=True,oldbox_corner_max_m=max(corner_errors),calibration_roundtrip_max_m=max(contract),
        blocked_canaries=blocked,rgb_bundle_files_read=sorted(opened),isolation='Network namespace plus Python open audit and sanitized copied RGB bundle; not a hostile-code native filesystem sandbox',metadata_contains_GT=False))
if __name__=='__main__':main()
