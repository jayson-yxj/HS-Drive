"""Actual synchronized sealed records and no cross-set duplicate RGB."""
import json
from pathlib import Path
import numpy as np
from tools.m5j_common import rows,write
from tools.m5f_common import input_path
from simulator.recorder import file_hash

def main():
    from tools.m5j_freeze import verify
    freeze=verify();rr=rows('test');old=rows('train')+rows('val')+rows('historical');oldhash={im['sha256']for r in old for im in json.loads(input_path(r).read_text())['images']};plan={g['group']:g for g in json.loads(Path('configs/m5j_capture_test.json').read_text())['groups']};records=[];newhash=set()
    for r in rr:
        raw=Path(r['raw_frame']).parent.parent;f=json.loads(Path(r['raw_frame']).read_text());man=json.loads((raw/'manifest.json').read_text());assert file_hash(raw/'calibration.json')==man['calibration_sha256'];xy=np.asarray(f['T_world_from_ego'])[:2,3]*[1,-1];error=float(np.linalg.norm(xy-plan[r['group']]['pose'][:2]));assert error<.5;checks=0
        for c,d in f['cameras'].items():
            for k in ['rgb','depth','instance']:
                assert d[k+'_frame_id']==f['frame_id'];assert abs(d[k+'_timestamp_s']-f['timestamp_s'])<1e-6;assert file_hash(raw/d[k])==d[k+'_sha256'];checks+=1
            newhash.add(d['rgb_sha256'])
        records.append(dict(group=r['group'],sample_id=r['sample_id'],configuration=r['configuration'],sensor_checks=checks,pose_xy_error_m=error,cameras=[dict(camera=c['camera'],actor_views=sum(v['pixels']>=3 for v in c['actors']))for c in r['actors']]))
    assert not newhash&oldhash;assert len(plan)==len({r['group']for r in rr})
    write('reports/m5j_dataset_quality.json',dict(kind='actual_CARLA_recorded_integrity_and_split_audit',records=records,passed=True,duplicate_historical_rgb=0,positions=plan,reference='selfrendererfirstsurface notindependentmesh; ActorAABB associationapproximate',source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
