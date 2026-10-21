"""Verify exact torchvision resize, including rounding and padded tensor size."""
from pathlib import Path
import json,torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from simulator.recorder import atomic_json,file_hash

def main():
    p=Path('reports/m5h_teacher_resize_verification.json')
    if p.exists():raise FileExistsError(p)
    t=GeneralizedRCNNTransform(800,1333,[.485,.456,.406],[.229,.224,.225]).eval();out,_=t([torch.zeros(3,384,640)]);cache=json.loads(Path('runs/m5h_teacher_trainval/index.json').read_text());atomic_json(p,dict(kind='actual_CPU_official_transform_verification',requested_min=800,requested_max=1333,native_hw=[384,640],actual_resized_hw=list(out.image_sizes[0]),padded_hw=list(out.tensors.shape[-2:]),source_sha256=file_hash(__file__),cache_index_sha256=file_hash('runs/m5h_teacher_trainval/index.json'),correction='Cached index resize.actual_hw [800,1333] was nominal not exact; actual interpolation rounds to799x1333 and pads800x1344. Predictions alreadyusedofficialtransform; no image/prediction change. Preserveinitialmetadatawiththiscorrection.'))
if __name__=='__main__':main()
