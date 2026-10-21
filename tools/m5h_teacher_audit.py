"""Teacher errors and agreement-mask audit; report observations, never rewrite GT."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tools.m5h_data import rows
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    dest=Path('reports/m5h_teacher_quality.json')
    if dest.exists():raise FileExistsError(dest)
    rr=rows();idx={r['sample_id']:r for r in json.loads(Path('runs/m5h_teacher_trainval/index.json').read_text())['records']};records=[];actors=[]
    for r in rr:
        d=pixel_label(r);q=np.load(idx[r['sample_id']]['path']);score=q['score'].astype(np.float32)/256;trusted=q['trusted'].astype(np.float32)/256;supported=np.isin(d['classes'],[1,2]);target=d['target']&supported;valid=d['valid']&(~d['target']|target);p=score.max(0);agree=np.where(d['classes']==1,trusted[0],trusted[1]);mask=valid&target&(agree>=.7)
        for c,cam in enumerate(r['camera_names']):
            for lo,hi in [(0,10),(10,20),(20,35),(35,80)]:
                m=valid[c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi);positive=m&target[c];negative=m&~target[c];records.append(dict(sample_id=r['sample_id'],split=r['split'],group=r['group'],camera=cam,weather=r['weather'],distance=[lo,hi],positive=int(positive.sum()),negative=int(negative.sum()),teacher_true_pixels=int((positive&(p[c]>=.7)).sum()),teacher_missed_pixels=int((positive&(p[c]<.7)).sum()),teacher_background_fp=int((negative&(p[c]>=.7)).sum()),distillation_pixels=int((m&mask[c]).sum()),unsupported_cone_pixels=int((d['target'][c]&(d['classes'][c]==3)).sum()),unknown_predicted_pixels=int((~d['valid'][c]&(p[c]>=.7)).sum())))
            for a in r['actors'][c]['actors']:
                m=(d['actor'][c]==a['local_id'])&target[c];n=int(m.sum());hit=int((m&(p[c]>=.7)).sum());actors.append(dict(sample_id=r['sample_id'],split=r['split'],group=r['group'],camera=cam,type_id=a['type_id'],distance=a['distance_m'],visible_pixels=n,hits=hit,detected=n>=3 and hit>=3 and hit/n>=.5))
    atomic_json(dest,dict(kind='offline_actual_train_validation_teacher_cache_audit',records=records,actors=actors,teacher_index_sha256=file_hash('runs/m5h_teacher_trainval/index.json'),source_sha256=file_hash(__file__),mask='GTvalid & supported GTpositive & sameclass instanceconfidence>=.7 & pixel mask*score>=.7; ALL teachernegatives excluded',limits='1px rendersemantic/instance boundaries ignored by GT; teacher responses there reportedUnknown, cannot label occlusion-edge FP without trustworthyboundaryGT. Small/far supportcounts shown. Coneunsupported; no negativeKD. No mask scores interpretedasphysicaloccupancy.'))
if __name__=='__main__':main()
