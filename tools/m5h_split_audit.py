"""Cross-stage geometry and checksum separation including original driving replay."""
import json
from pathlib import Path
import numpy as np
from tools.m5h_data import rows
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5h_split_audit_final.json')
    if out.exists():raise FileExistsError(out)
    rr=rows();poses={};hashes={}
    for r in rr:
        raw=Path(r['raw_frame']);f=json.loads(raw.read_text());p=np.asarray(f['T_world_from_ego'])[:2,3]*[1,-1];poses[r['group']]=dict(map=r['map'].split('/')[-1],split=r['split'],xy=p.tolist());inp=json.loads(Path(r['input']).read_text());hashes[r['sample_id']]=[x['sha256']for x in inp['images']]
    man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());ix=json.loads(Path(man['config']['index'],'index.json').read_text());replay=[s for s in ix['scenes']if s['split']=='train'];tests=json.loads(Path('configs/m5h_capture_test.json').read_text())['groups'];dist=[]
    for t in tests:
        sources=[dict(group=g,points=[p['xy']],scope='pixel_'+p['split'])for g,p in poses.items()if p['map']==t['map']]+[dict(group=s['scene_id'],points=s['positions_xy'],scope='M4_replay_train')for s in replay if s['map']==t['map']]
        d=[dict(source=s['group'],scope=s['scope'],distance_m=float(np.linalg.norm(np.array(s['points'])-t['pose'][:2],axis=1).min()))for s in sources];dist.append(dict(test_group=t['group'],sources=d,min_distance_m=min([x['distance_m']for x in d]or[1000.])))
    train={h for r in rr if r['split']=='train'for h in hashes[r['sample_id']]};val={h for r in rr if r['split']=='val'for h in hashes[r['sample_id']]};atomic_json(out,dict(kind='offline_geometry_only_pretest_audit',test_pose_distances=dist,cross_train_val_rgb_identical=len(train&val),positions=poses,source_sha256=file_hash(__file__),limits='Scene separation does not imply newassets/layouttemplates; replay routepoints included, HS recordedworld y reflectedbacktoCARLA; protocolandreplaypositions useCARLA. Prior initial/corrected audits superseded due mapnameandhandednessbugs.'))
    if train&val or any(d['min_distance_m']<45 for d in dist):raise ValueError('Separation failed; beforecapture choose newgeometry')
if __name__=='__main__':main()
