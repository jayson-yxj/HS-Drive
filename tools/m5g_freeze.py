"""Freeze model choices using only validation, before new test acquisition."""
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    path=Path('reports/m5g_pretest_freeze.json')
    if path.exists():raise FileExistsError(path)
    pixel=json.loads(Path('reports/m5g_pixel_val.json').read_text());depth=json.loads(Path('reports/m5g_depth_training.json').read_text());pre=json.loads(Path('reports/m5g_pretrained_val.json').read_text());dv=json.loads(Path('reports/m5g_depth_val.json').read_text());choices={}
    training=json.loads(Path('reports/m5g_backbone_ablation.json').read_text())['records']+json.loads(Path('reports/m5g_budget_extension.json').read_text())['records']
    for seed in [101,102]:
        candidates=[r for r in training if r['seed']==seed];best=min(candidates,key=lambda r:(r['selected']['violation'],-r['selected']['macro_iou'],r['selected']['step'],r['name']));ds=[r for r in depth['records']if r['seed']==seed];db=max(ds,key=lambda r:next(h for h in r['history']if h['step']==1200 and 'validation'in h)['validation']['target_spatial_macro']);choices[str(seed)]=dict(pixel=best['name'],pixel_checkpoint=best['checkpoint'],pixel_sha256=best['sha256'],depth=db['name'],depth_checkpoint=f"runs/m5g_depth/m5g_{db['name']}_{seed}_step1200.pt",depth_sha256=file_hash(f"runs/m5g_depth/m5g_{db['name']}_{seed}_step1200.pt"),depth_release_candidate='original_only_neargate_rejects_alltrained',depth_exploratory=True)
    checkpoints={r['checkpoint']:r['sha256']for r in training};checkpoints.update({r['selected']['checkpoint']:r['selected']['sha256']for r in depth['records']});checkpoints.update({v['depth_checkpoint']:v['depth_sha256']for v in choices.values()});sources={str(p):file_hash(p)for d in ['tools','training','tests','configs']for p in Path(d).glob('*m5g*')if p.is_file()}
    atomic_json(path,dict(kind='validation_only_scientific_freeze_before_new_capture',choices=choices,checkpoints=checkpoints,sources=sources,protocol_sha256=file_hash('configs/m5g_protocol.json'),pretrained_weight_sha256=pre['weight_sha256'],decision='Allmatchedarmsreported; selectedcandidateonlyforshadow. No newtraining/reselection afternewtest. 2400budget separatefrom1200architecturecomparison.',source_sha256=file_hash(__file__)))
    print(choices,flush=True)

if __name__=='__main__':main()
