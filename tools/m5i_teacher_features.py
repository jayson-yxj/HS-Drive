"""Offline FPN feature transfer cache with explicit resize/conv-grid alignment."""
import json,time
from pathlib import Path
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from models.m5i_visual import align_teacher
from tools.m5i_data import rows
from tools.m5f_common import inputs
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    out=Path('runs/m5i_teacher_features');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);path='/root/gpufree-data/hs-drive-cache/weights/m5g/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth';net=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,box_score_thresh=.05).cuda().eval();net.load_state_dict(torch.load(path,weights_only=True));records=[];start=time.perf_counter()
    for row in rows():
        x=inputs(row);features=[];meta=[]
        with torch.no_grad():
            for c in range(6):
                im,_=net.transform([x['images'][0,c]]);f=net.backbone(im.tensors)['1'];aligned=align_teacher(f,(384,640),im.image_sizes[0]);features.append(aligned[0].to(torch.bfloat16).cpu());meta.append(dict(native_hw=[384,640],resized_hw=list(im.image_sizes[0]),padded_hw=list(im.tensors.shape[-2:]),feature_hw=list(f.shape[-2:]),channel=256,key='1',stride=8))
        p=out/(row['sample_id'].replace('/','_')+'.pt');torch.save(dict(feature=torch.stack(features)),p);records.append(dict(sample_id=row['sample_id'],path=str(p),sha256=file_hash(p),geometry=meta));print(row['sample_id'],flush=True)
    atomic_json(out/'index.json',dict(kind='actual_GPU_offline_RGB_only_teacher_FPN',records=records,bytes=sum(Path(r['path']).stat().st_size for r in records),elapsed_s=time.perf_counter()-start,teacher_weight_sha256=file_hash(path),source_sha256=file_hash(__file__),alignment_source_sha256=file_hash('models/m5i_visual.py'),definition='Studentconvcentersnative8j -> resized(u+.5)*scale-.5 -> teacherconvindex/8 -> grid_samplealign_cornersFalse. Actualresized799x1333 excludespadding. BF16storage FP32normalization; adapterdiscardedatinference.'))
if __name__=='__main__':main()
