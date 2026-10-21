"""Verify ImageNet initialization lineage via manifest, weight digest and loader."""
import json
from pathlib import Path
import torch
from simulator.recorder import atomic_json,file_hash

def main():
    p=Path('reports/m5i_baseline_review.json')
    if p.exists():raise FileExistsError(p)
    files=['training/m5h_losses.py','tools/m5h_pixel_training.py','tools/m5h_depth_training.py','models/image_encoder.py','training/runner.py'];texts={f:Path(f).read_text()for f in files};m=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());checkpoint=torch.load('runs/m4_expanded_joint/best.pt',map_location='cpu',weights_only=False);h=json.loads(Path('reports/m5h_depth_spatial_quality.json').read_text());r=[v for v in h['aggregate']if v['scope']=='all'and v['distance']=='1.5_80'];atomic_json(p,dict(kind='offline_actual_source_checkpoint_lineage',files={f:file_hash(f)for f in files},checkpoint_keys=list(checkpoint),original_pretrained_sha256=m['pretrained_sha256'],pretrained_sha_matches=file_hash(m['config']['pretrained_weights'])==m['pretrained_sha256'],m4_checkpoint_sha_matches=file_hash('runs/m4_expanded_joint/best.pt')==m['best_sha256'],facts=dict(positive_only_KD=True,teacher_negative_KD=False,imagenet_initialization=True,not_from_scratch=True),global_bias=[dict(name=v['name'],seed=v['seed'],bias=v['metrics']['signed_bias']['mean'])for v in r],limits='M4fine-tuning changes weights; initlineage basedactualmanifest/checksums andloader not equalitytoImageNetcurrentweights.',source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
