"""Class-conditional teacher errors, never used to overwrite GT."""
import json
from pathlib import Path
import numpy as np
from tools.m5i_data import rows
from tools.m5i_teacher_audit import teacher_index
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_teacher_class_reliability.json')
    if out.exists():raise FileExistsError(out)
    idx=teacher_index();records=[]
    for row in rows():
        d=pixel_label(row);q=np.load(idx[row['sample_id']]['path']);score=q['score'].astype(np.float32)/256;trusted=q['trusted'].astype(np.float32)/256
        for c,cam in enumerate(row['camera_names']):
            for lo,hi in [(0,5),(5,10),(10,20),(20,35),(35,80)]:
                area=d['valid'][c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi)
                for k in [1,2]:
                    pos=area&d['target'][c]&(d['classes'][c]==k);n=int(pos.sum());sc=score[k-1,c];ts=trusted[k-1,c]
                    records.append(dict(group=row['group'],sample_id=row['sample_id'],split=row['split'],camera=cam,distance=[lo,hi],class_id=k,positive_n=n,positive_teacher_agreement=int((pos&(ts>=.7)).sum()),miss_low_score=int((pos&(sc<.05)).sum()),mean_score=float(sc[pos].mean())if n else None,gt_negative_high_score=int((area&~d['target'][c]&(sc>=.7)).sum())))
    atomic_json(out,dict(kind='offline_CARLA_GT_and_offline_RGB_teacher_class_audit',records=records,source_sha256=file_hash(__file__),limits='Vehicle/person only; cones unsupported. WrongclassGTforeground is notbackgroundnegative. All correlated pixels; no independent meshes.'))
if __name__=='__main__':main()
