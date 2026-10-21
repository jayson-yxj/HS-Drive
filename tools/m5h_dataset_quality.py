"""Independent-location count and measured test support, never model selection."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m5h_data import rows
from tools.m5h_shadow import TEST
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5h_dataset_quality.json')
    if out.exists():raise FileExistsError(out)
    rr=rows(TEST);records=[]
    for r in rr:
        d=pixel_label(r)
        for c,cam in enumerate(r['camera_names']):
            for k,name in [(1,'vehicle'),(2,'pedestrian'),(3,'cone')]:
                t=d['target'][c]&(d['classes'][c]==k);records.append(dict(group=r['group'],sample_id=r['sample_id'],configuration=r['configuration'],camera=cam,weather=r['weather'],class_name=name,positive_pixels=int(t.sum()),distance={f'{lo}_{hi}':int((t&(d['radial'][c]>=lo)&(d['radial'][c]<hi)).sum())for lo,hi in [(0,10),(10,20),(20,35),(35,80)]},confirmed_negative_pixels=int((d['valid'][c]&~d['target'][c]).sum())))
    atomic_json(out,dict(kind='offline_audit_actual_synchronized_CARLA_test_capture',frames=len(rr),independent_location_groups=len({r['group']for r in rr}),records=records,source_sha256=file_hash(__file__),index_sha256=file_hash(Path(TEST,'index.json')),limits='12layoutvariants/location correlated; sameassets/layouttemplates, no Meshphysicaltruth. Ego-removedrenderreference supports selfmask but not independenttruth. First surface semantics andconservativeActorassociation only.'))
if __name__=='__main__':main()
