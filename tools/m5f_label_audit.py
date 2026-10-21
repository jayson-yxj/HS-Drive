"""Supplementary sealed-test audit; no rule changes and no training selection."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_test_label_quality.json')
    if dest.exists():raise FileExistsError(dest)
    root=Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels');index=json.loads((root/'index.json').read_text());quality=json.loads((root/'quality.json').read_text());groups=[];sync=[]
    for group in sorted({r['group'] for r in index['records']}):
        rows=[r for r in index['records'] if r['group']==group];stats=[s for r in quality['records'] if r['group']==group for s in r['cameras']];actors=[a for r in rows for c in r['actors'] for a in c['actors']];types={}
        for typ in sorted({a['type_id'] for a in actors}):
            aa=[a for a in actors if a['type_id']==typ];types[typ]=dict(camera_observations=len(aa),eligible=sum(a['pixels']>=3 for a in aa),pixels=sum(a['pixels'] for a in aa),zero_support=sum(a['pixels']==0 for a in aa))
        calhash=[];poses=[]
        for row in rows:
            frame=json.loads(Path(row['raw_frame']).read_text());raw=Path(row['raw_frame']).parent.parent;calhash.append(file_hash(raw/'calibration.json'));poses.append(frame['T_world_from_ego']);entries=[e for e in frame['cameras'].values()];checks=[e[k+'_frame_id']==frame['frame_id'] and abs(e[k+'_timestamp_s']-frame['timestamp_s'])<1e-5 for e in entries for k in ['rgb','depth','instance']];sync.append(dict(sample_id=row['sample_id'],sensor_stamps=len(checks),all_matched=all(checks)))
        groups.append(dict(group=group,frames=len(rows),positive=sum(s['positive'] for s in stats),negative=sum(s['negative'] for s in stats),valid_fraction=sum(s['valid'] for s in stats)/(len(stats)*384*640),types=types,calibration_identical=len(set(calhash))==1,ego_pose_identical=bool(np.allclose(poses,np.array(poses[0])[None],atol=1e-6)),mean_boundary_near_rgbedge=float(np.mean([s['boundary_near_rgb_edge_fraction'] for s in stats if s['boundary_near_rgb_edge_fraction'] is not None]))))
    freeze=json.loads(Path('reports/m5f_pretest_decision.json').read_text());checks={p:file_hash(p)==sha for p,sha in freeze['sources'].items()};atomic_json(dest,dict(kind='offline_audit_actual_new_CARLA_capture',groups=groups,synchronization=sync,frozen_scientific_sources_unchanged=checks,checkpoint_unchanged={p:file_hash(p)==sha for p,sha in freeze['checkpoints'].items()},source_index_sha256=file_hash(root/'index.json'),quality_sha256=file_hash(root/'quality.json'),source_sha256=file_hash(__file__),limits='36frames3groups; reportzero-support andsemanticassociationloss, cannot claim everyplacedActor visible. Contemporaneoussensor metadata verified; RGBedge onlyalignmentproxy. Transparent/reflection interiors unresolved. Metadata notmodelinputs.'))

if __name__=='__main__':main()
