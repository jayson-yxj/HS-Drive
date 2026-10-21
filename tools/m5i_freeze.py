"""Validation-only model and source freeze before unseen test acquisition."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def verify():
    f=json.loads(Path('reports/m5i_pretest_freeze.json').read_text())
    for key in ['sources','checkpoints','selection_reports']:
        for p,h in f[key].items():
            if file_hash(p)!=h:raise ValueError('Frozen asset changed '+p)
    return f

def main():
    dest=Path('reports/m5i_pretest_freeze.json')
    if dest.exists():raise FileExistsError(dest)
    study=json.loads(Path('reports/m5i_distillation_comparison.json').read_text())['records'];extra=Path('reports/m5i_distillation_comparison_feature.json')
    if extra.exists():study+=json.loads(extra.read_text())['records']
    depth=json.loads(Path('reports/m5h_pretest_freeze.json').read_text())['choices'];choices={};ck={r['checkpoint']:r['sha256']for r in study}
    for seed in [101,102]:
        best=min([r for r in study if r['seed']==seed],key=lambda r:(r['selected']['violation'],-r['selected']['macro_iou'],r['selected']['step'],r['name']));d=depth[str(seed)];choices[str(seed)]=dict(pixel=best['name'],pixel_checkpoint=best['checkpoint'],pixel_sha256=best['sha256'],pixel_step=best['selected']['step'],pixel_threshold=best['selected']['threshold'],pixel_joint_qualified=best['selected']['violation']<=0,depth='replay',depth_checkpoint=d['depth_checkpoint'],depth_sha256=d['depth_sha256']);ck[d['depth_checkpoint']]=d['depth_sha256']
    sources={str(p):file_hash(p)for root in ['tools','models','training','tests','configs','diagnostics']for p in Path(root).glob('*m5i*')if p.is_file()};reports={str(p):file_hash(p)for p in Path('reports').glob('m5i*.json')};atomic_json(dest,dict(kind='actual_pretest_validation_only_freeze',choices=choices,checkpoints=ck,sources=sources,selection_reports=reports,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),no_more_training=True,source_sha256=file_hash(__file__)))
    print(choices)
if __name__=='__main__':main()
