"""Runs only inside the network-disabled PETR sandbox; no project model import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

OUT=Path(os.environ.get('M8A_OUTPUT','/output'))
def save(name,data):
    with (OUT/(name+'.json')).open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False)

def main():
    assert os.environ.get('M8A_ISOLATED')=='1'
    start=time.perf_counter()
    import torch
    import numpy as np
    torch.manual_seed(8021)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    t=time.perf_counter();torch.cuda.init();torch.cuda.synchronize()
    device=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,runtime=torch.version.cuda,arch_list=torch.cuda.get_arch_list(),
        cuda_initialization_ms=1000*(time.perf_counter()-t))
    a=torch.randn(128,128,device='cuda');b=a@a.T;torch.cuda.synchronize()
    assert torch.isfinite(b).all()
    save('torch_cuda',dict(status='Passed',device=device,input_shape=list(a.shape),dtype=str(a.dtype),output_finite=True))
    import mmcv
    from mmcv.ops import ModulatedDeformConv2dPack, get_compiler_version, get_compiling_cuda_version
    import mmcv._ext as ext
    dcn=ModulatedDeformConv2dPack(8,16,3,padding=1).cuda().eval()
    x=torch.randn(2,8,32,32,device='cuda')
    with torch.no_grad():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            y=dcn(x);torch.cuda.synchronize()
        repeat=[dcn(x) for _ in range(3)];torch.cuda.synchronize()
    kernels=sorted({event.name for event in prof.events() if event.device_type==torch.autograd.DeviceType.CUDA})
    assert torch.isfinite(y).all() and all(torch.equal(y,v) for v in repeat)
    assert any('deformable' in k.lower() or 'deform' in k.lower() for k in kernels),kernels
    prof.export_chrome_trace(str(OUT/'dcn_cuda_trace.json'))
    save('cuda_operators',dict(status='Passed',device=device,mmcv=mmcv.__version__,compiler=get_compiler_version(),
        compiling_cuda=get_compiling_cuda_version(),extension=ext.__file__,extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        input_shape=list(x.shape),output_shape=list(y.shape),dtype=str(y.dtype),cuda_kernels=kernels,
        repeated_bitwise_equal=True,all_finite=True,fallback=False))
    del dcn,x,y,repeat,a,b
    import sys
    sys.path.insert(0,os.environ['PETR_SOURCE'])
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    cfg=mmcv.Config.fromfile(os.environ['PETR_SOURCE']+'/projects/configs/petr/petr_r50dcn_gridmask_c5.py')
    previous=cfg.model.img_backbone.pretrained
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    safe_file=Path(os.environ['M8A_SAFE_WEIGHTS'])
    assert hashlib.sha256(safe_file.read_bytes()).hexdigest()==os.environ['M8A_SAFE_SHA256']
    with np.load(safe_file,allow_pickle=False) as arrays:
        state={key:torch.from_numpy(arrays[key].copy()) for key in arrays.files}
    expected=model.state_dict()
    missing=sorted(set(expected)-set(state));unexpected=sorted(set(state)-set(expected))
    comparisons=[dict(key=k,model_shape=list(v.shape),checkpoint_shape=list(state[k].shape) if k in state else None,
                      shape_equal=k in state and v.shape==state[k].shape) for k,v in expected.items()]
    mismatches=[v for v in comparisons if not v['shape_equal']]
    save('checkpoint_key_audit',dict(missing=missing,unexpected=unexpected,shape_mismatches=mismatches,keys=comparisons,
        model_tensor_count=len(expected),checkpoint_tensor_count=len(state)))
    assert not missing and not unexpected and not mismatches
    model.load_state_dict(state,strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    assert model.pts_bbox_head.num_query==900
    assert model.pts_bbox_head.num_classes==10
    assert len(model.pts_bbox_head.transformer.decoder.layers)==6
    dcn_names=[name for name,module in model.named_modules() if isinstance(module,ModulatedDeformConv2dPack)]
    assert len(dcn_names)==9,dcn_names
    save('model_load',dict(status='Passed',strict=True,all_loaded_tensors_bitwise_equal=True,parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint_tensor_count=len(state),dcn_modules=dcn_names,num_query=900,decoder_layers=6,classes=10,
        configuration_changes={'img_backbone.pretrained':{'before':previous,'after':None,'reason':'All final backbone tensors strictly loaded and compared'}}))
    del state,expected
    model.cuda().eval()
    # Proper right-handed synthetic rig: x-forward/y-left/z-up, optical x-right/y-down/z-forward.
    # Original nuScenes-sized synthetic BGR pixels, official deterministic resize/crop and normalization.
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage, NormalizeMultiviewImage, PadMultiViewImage
    rng=np.random.RandomState(8021)
    intrinsic=np.eye(4);intrinsic[:3,:3]=[[1000,0,800],[0,1000,450],[0,0,1]]
    extrinsics=[]
    for angle in [0,60,-60,120,-120,180]:
        rad=np.deg2rad(angle);c,s=np.cos(rad),np.sin(rad)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        c2e=np.eye(4);c2e[:3,:3]=rot@np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        c2e[:3,3]=[1.2*c,1.2*s,1.6]
        extrinsics.append(np.linalg.inv(c2e).T)
    data=dict(img=[rng.randint(0,256,(900,1600,3),dtype=np.uint8) for _ in range(6)],
              intrinsics=[intrinsic.copy() for _ in range(6)],extrinsics=extrinsics)
    data=ResizeCropFlipImage(cfg.ida_aug_conf,training=False)(data)
    data=NormalizeMultiviewImage(**cfg.img_norm_cfg)(data)
    data=PadMultiViewImage(size_divisor=32)(data)
    cpu=np.stack(data['img']).transpose(0,3,1,2).copy()[None]
    assert list(cpu.shape)==[1,6,3,512,1408]
    inp=torch.from_numpy(cpu).cuda()
    meta=dict(img_shape=[(512,1408,3)]*6,pad_shape=[(512,1408,3)]*6,ori_shape=[(900,1600,3)]*6,
              lidar2img=data['lidar2img'],box_type_3d=LiDARInstance3DBoxes,scale_factor=1.,flip=False)
    save('synthetic_input',dict(kind='SyntheticInputSmoke_NotNuScenesEvaluation',shape=list(cpu.shape),dtype=str(cpu.dtype),
        tensor_sha256=hashlib.sha256(cpu.tobytes()).hexdigest(),angles_deg=[0,60,-60,120,-120,180],
        lidar2img=[v.tolist() for v in data['lidar2img']],img_norm_cfg=dict(cfg.img_norm_cfg),
        note='Synthetic calibrated ring, not official nuScenes camera poses or detection quality evidence'))
    shape_trace={}
    def trace(name):
        def hook(module,args,output):
            if isinstance(output,dict):
                shape_trace[name]={k:list(v.shape) for k,v in output.items() if torch.is_tensor(v)}
                assert all(torch.isfinite(v).all() for v in output.values() if torch.is_tensor(v))
        return hook
    handle=model.pts_bbox_head.register_forward_hook(trace('head'))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    def forward():
        # Official extract_img_feat squeezes input in-place: fresh view for every call.
        with torch.no_grad():return model(return_loss=False,rescale=True,img=[inp.view(1,6,3,512,1408)],img_metas=[[dict(meta)]])
    timings=[];outputs=[]
    for index in range(24):
        torch.cuda.synchronize();t=time.perf_counter();result=forward();torch.cuda.synchronize()
        ms=1000*(time.perf_counter()-t);timings.append(ms)
        obj=result[0]['pts_bbox'];box=obj['boxes_3d'].tensor;scores=obj['scores_3d'];labels=obj['labels_3d']
        assert torch.isfinite(box).all() and torch.isfinite(scores).all()
        outputs.append(dict(iteration=index,boxes=box.cpu().tolist(),scores=scores.cpu().tolist(),labels=labels.cpu().tolist(),shape=list(box.shape)))
        print('forward',index,'ms',round(ms,3),'boxes',list(box.shape),flush=True)
        if index==0:handle.remove()
    measured=timings[4:]
    save('forward_results',dict(status='Passed',kind='ActualRTX4090_SyntheticInputSmoke',outputs=outputs,head_shape_trace=shape_trace,
        input_shape=[1,6,3,512,1408],precision='FP32_no_autocast_no_TF32',all_finite=True,training_steps=0,quality_validated=False))
    save('gpu_memory_latency',dict(status='Passed',cold_forward_ms=timings[0],warmup_ms=timings[1:4],measured_ms=measured,
        p50_ms=float(np.percentile(measured,50)),p95_ms=float(np.percentile(measured,95)),max_ms=max(measured),
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_device_total_bytes=torch.cuda.get_device_properties(0).total_memory,process_gpu_memory='External nvidia-smi monitor; includes context',
        elapsed_s=time.perf_counter()-start,p99_certified=False,shared_gpu_20hz_certified=False))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('failure',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
