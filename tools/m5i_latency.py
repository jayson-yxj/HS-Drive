"""Standalone six-camera latency after CARLA and GPU experiments stop."""
import json,time,subprocess
from pathlib import Path
import torch,numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from tools.m5i_data import load
from tools.m5f_common import inputs
from tools.m5a_learning import encoder_load
from models.m5f_pixel import ForegroundModel
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5f_lift import lift
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    path=Path('reports/m5i_inference_latency.json')
    if path.exists():raise FileExistsError(path)
    from tools.m5i_freeze import verify
    verify()
    prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5i_test_pixels/index.json').read_text())['records'];x=inputs(rows[2]);study=json.loads(Path('reports/m5i_pixel_test.json').read_text())['records'];records=[];snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    def bench(name,call,steps=20):
        torch.cuda.reset_peak_memory_stats();times=[]
        for i in range(steps+5):
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():call()
            torch.cuda.synchronize()
            if i>=5:times.append((time.perf_counter()-start)*1000)
        records.append(dict(name=name,times_ms=times,p50=float(np.median(times)),p95=float(np.quantile(times,.95)),max=float(max(times)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    for r in study:
        model,ck=load(dict(name=r['name'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']))
        def pixel():
            with torch.autocast('cuda',dtype=torch.bfloat16):return model(**x)
        bench(r['name']+'_'+str(r['seed']),pixel);del model
    frozen=json.loads(Path('reports/m5i_pretest_freeze.json').read_text())
    for seed,c in frozen['choices'].items():
        pixel,ck=load(dict(name=c['pixel'],checkpoint=c['pixel_checkpoint'],sha256=c['pixel_sha256']));net=encoder_load();net.depth_head.load_state_dict(torch.load(c['depth_checkpoint'],weights_only=True)['depth'])
        def joint():
            with torch.autocast('cuda',dtype=torch.bfloat16):p=pixel(**x).float().sigmoid();dc=cache_p8(net,x);return lift(net.splat,p[:,:,4::8,4::8],dc['probs'],dc['K'],dc['T'],dc['image_hw'])
        bench('joint_'+seed,joint);del pixel,net
    net=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,box_score_thresh=.05);net.load_state_dict(torch.load('/root/gpufree-data/hs-drive-cache/weights/m5g/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth',weights_only=True));net.cuda().eval();bench('COCO_MaskRCNN_sequential6',lambda:[net([x['images'][0,c]])for c in range(6)],10)
    atomic_json(path,dict(kind='actual_isolated_RTX4090_recorded6RGB_timing',records=records,gpu=torch.cuda.get_device_name(),gpu_process_snapshot=snapshot,source_sha256=file_hash(__file__),limits='ExcludesI/O/render/control; FP32COCO default800resize vsBF16HS384x640. No strictrealtimeclaim. CacheP8computesonlyfrozenRGB noGT.'))

if __name__=='__main__':main()
