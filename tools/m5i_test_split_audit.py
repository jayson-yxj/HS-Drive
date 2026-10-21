"""Actual held-out capture integrity versus predeclared geometry and train/val RGB."""
import json
from pathlib import Path
import numpy as np
from tools.m5i_data import rows,TEST
from tools.m5f_common import input_path
from tools.m5i_freeze import verify
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_test_split_audit.json')
    if out.exists():raise FileExistsError(out)
    verify();old=rows();new=rows(TEST);plan={r['group']:r for r in json.loads(Path('configs/m5i_capture_test.json').read_text())['groups']};hashes={};records=[]
    for split,rr in [('trainval',old),('test',new)]:
        hashes[split]=set()
        for r in rr:
            inp=json.loads(input_path(r).read_text());hashes[split].update(x['sha256']for x in inp['images'])
            if split=='test':
                f=json.loads(Path(r['raw_frame']).read_text());xy=np.asarray(f['T_world_from_ego'])[:2,3]*[1,-1];error=float(np.linalg.norm(xy-plan[r['group']]['pose'][:2]));assert error<.5
                records.append(dict(group=r['group'],sample_id=r['sample_id'],pose_xy_error_m=error,map=r['map'],split=r['split']))
    duplicate=hashes['trainval']&hashes['test'];assert not duplicate;assert {r['group']for r in new}==set(plan)
    atomic_json(out,dict(kind='actual_CARLA_heldout_pose_RGB_integrity',records=records,duplicate_rgb_hashes=len(duplicate),passed=True,source_sha256=file_hash(__file__),limits='Independent locations, shared actorassets/layouts; not fully new scene distributions. Plan45m isolation verified separately.'))
if __name__=='__main__':main()
