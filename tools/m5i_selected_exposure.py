"""Audit selected and final checkpoints; final exposure is not early-model exposure."""
import json
from pathlib import Path
from collections import Counter
from tools.m5i_data import rows
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_selected_exposure.json')
    if out.exists():raise FileExistsError(out)
    rr=rows();labels=[pixel_label(r)for r in rr];study=json.loads(Path('reports/m5i_distillation_comparison.json').read_text())['records'];extra=Path('reports/m5i_distillation_comparison_feature.json')
    if extra.exists():study+=json.loads(extra.read_text())['records']
    schedules={r['seed']:r['schedule']for r in json.loads(Path('reports/m5i_training_exposure.json').read_text())['records']};records=[];matching=[]
    for seed in [101,102]:
        ab=[r for r in study if r['seed']==seed and r['name']in['gt','output_kd']]
        assert len({r['initialization_sha256']for r in ab})==1
        assert len({r['schedule_sha256']for r in ab})==1
        matching.append(dict(seed=seed,initialization_sha256=ab[0]['initialization_sha256'],schedule_sha256=ab[0]['schedule_sha256'],passed=True))
    for record in study:
        for checkpoint,step in [('selected',record['selected']['step']),('final',record['steps'])]:
            seen=set(tuple(v)for v in schedules[record['seed']][:step]);cams=[]
            for c,name in enumerate(rr[0]['camera_names']):
                views=[i for i,cam in seen if cam==c];positive=[i for i in views if labels[i]['target'][c].any()];cams.append(dict(camera=name,unique_views=len(views),positive_views=len(positive),positive_groups=sorted({rr[i]['group']for i in positive})))
            records.append(dict(name=record['name'],seed=record['seed'],checkpoint=checkpoint,step=step,unique_views=len(seen),cameras=cams))
    atomic_json(out,dict(kind='actual_checkpoint_training_schedule_audit',records=records,matched_AB=matching,source_sha256=file_hash(__file__),limits='Positiveview here anyreliabletargetpixel; stricterassociatedActor-view coverage independentlyreported.'))
if __name__=='__main__':main()
