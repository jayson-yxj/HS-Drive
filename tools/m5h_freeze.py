"""Validation-only selection and immutable test opening boundary."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def verify():
    f=json.loads(Path('reports/m5h_pretest_freeze.json').read_text())
    for key in ['checkpoints','sources','selection_reports']:
        for p,h in f[key].items():
            if file_hash(p)!=h:raise ValueError('Frozen asset changed: '+p)
    return f

def main():
    dest=Path('reports/m5h_pretest_freeze.json')
    if dest.exists():raise FileExistsError(dest)
    pixel=json.loads(Path('reports/m5h_distillation_ablation.json').read_text());depth=json.loads(Path('reports/m5h_depth_preservation.json').read_text());choices={};checkpoints={}
    for r in pixel['records']:checkpoints[r['checkpoint']]=r['sha256']
    for r in depth['records']:checkpoints[r['selected']['checkpoint']]=r['selected']['sha256']
    for seed in [101,102]:
        ps=[r for r in pixel['records']if r['seed']==seed];p=min(ps,key=lambda r:(r['selected']['violation'],-r['selected']['macro_iou'],r['selected']['step'],r['name']));ds=[r for r in depth['records']if r['seed']==seed and r['selected']['step']>0 and r['selected']['near_gate']];d=max(ds,key=lambda r:r['selected']['validation']['target_spatial_macro'])if ds else None;choices[str(seed)]=dict(pixel=p['name'],pixel_checkpoint=p['checkpoint'],pixel_sha256=p['sha256'],pixel_threshold=p['selected']['threshold'],pixel_step=p['selected']['step'],depth=d['name']if d else'original',depth_checkpoint=d['selected']['checkpoint']if d else None,depth_sha256=d['selected']['sha256']if d else None,depth_step=d['selected']['step']if d else 0,depth_validation_eligible=d is not None)
    sources={str(p):file_hash(p)for folder in ['tools','training','diagnostics','tests','configs']for p in Path(folder).glob('*m5h*')if p.is_file()};selection_reports={str(p):file_hash(p)for p in Path('reports').glob('m5h*.json')};atomic_json(dest,dict(kind='validation_only_freeze_beforetestcapture',choices=choices,checkpoints=checkpoints,sources=sources,selection_reports=selection_reports,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=file_hash(__file__),decision='AllGT/distill/Deptharmsreported; choicesonlyshadow. No hardnegative/backboneexpansion absentstableKDvalidationgain. SpatialD optional deferred: evaluate nearpreservation generalization first; noautomatic extra training. No test-driven changes.'))
    print(choices)
if __name__=='__main__':main()
