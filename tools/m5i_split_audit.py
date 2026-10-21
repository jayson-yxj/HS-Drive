"""One coordinate contract for actual records, planned poses and M4 replay routes."""
import json
from pathlib import Path
import numpy as np
from tools.m5i_data import rows
from tools.m5f_common import input_path
from simulator.recorder import atomic_json,file_hash

def main():
    p=Path('reports/m5i_split_audit.json')
    if p.exists():raise FileExistsError(p)
    rr=rows();groups={};hashes={}
    for r in rr:
        f=json.loads(Path(r['raw_frame']).read_text());groups[r['group']]=dict(map=r['map'].split('/')[-1],split=r['split'],xy=(np.array(f['T_world_from_ego'])[:2,3]*[1,-1]).tolist());inp=json.loads(input_path(r).read_text());hashes.setdefault(r['split'],set()).update(x['sha256']for x in inp['images'])
    man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());ix=json.loads(Path(man['config']['index'],'index.json').read_text());tests=json.loads(Path('configs/m5i_capture_test.json').read_text())['groups'];dist=[]
    for t in tests:
        sources=[dict(name=g,points=[v['xy']])for g,v in groups.items()if v['map']==t['map']]+[dict(name=v['scene_id'],points=(np.array(v['positions_xy'])*[1,-1]).tolist())for v in ix['scenes']if v['map'].split('/')[-1]==t['map']];ds=[dict(name=s['name'],distance_m=float(np.linalg.norm(np.array(s['points'])-t['pose'][:2],axis=1).min()))for s in sources];dist.append(dict(test=t['group'],minimum_m=min([x['distance_m']for x in ds]or[1000.]),sources=ds))
    assert not hashes['train']&hashes['val'];assert all(x['minimum_m']>=45 for x in dist)
    atomic_json(p,dict(kind='actual_recorded_pixel_and_M4_index_geometry_audit',groups=groups,test_distances=dist,duplicate_train_val_RGB=0,source_sha256=file_hash(__file__),coordinate='All comparisonsCARLA; raw T_world_from_ego and M4positions_xy areHS -> multiply[1,-1]; canonicalmapbasename',limits='Positionseparation notnewlayout/assets. M5Htest excludedtrainval, captureprotocolincludesallhistoricposes.'))
if __name__=='__main__':main()
