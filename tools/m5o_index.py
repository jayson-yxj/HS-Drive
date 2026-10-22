"""Minimal target-only RGB index: no occupancy regeneration or inference GT fields."""
import argparse,json
from pathlib import Path
from tools.m5o_common import write,asset,provenance
from datasets.m5n_reference import synchronized
from simulator.recorder import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output);dest.mkdir(exist_ok=False,parents=True);samples=[];audits=[]
    for path in sorted(Path(a.raw).glob('*/manifest.json')):
        root=path.parent;man=json.loads(path.read_text());cal=json.loads((root/'calibration.json').read_text())['cameras'];names=list(cal)
        assert file_hash(root/'calibration.json')==man['calibration_sha256']
        for fid in man['frames']:
            fp=root/'frames'/f'{fid:08d}.json';f=json.loads(fp.read_text());ref=json.loads((root/'frames'/f"{man['references'][str(fid)]:08d}.json").read_text());byid={v['actor_id']:v for v in ref['actors']}
            import numpy as np
            delta=max([float(np.max(abs(np.array(v['world_matrix'])-byid[v['actor_id']]['world_matrix']))) for v in f['actors'] if not v['is_ego']] or [0])
            assert delta<.001
            for actor in f['actors']:
                assert actor['actor_snapshot_present'] and actor['actor_snapshot_frame_id']==fid
                assert abs(actor['actor_snapshot_timestamp_s']-f['timestamp_s'])<1e-5
            sample=f"{man['group']}/{fid:08d}";inp=dict(sample_id=sample,images=[dict(path=str(root/f['cameras'][n]['rgb']),sha256=f['cameras'][n]['rgb_sha256']) for n in names],intrinsics=[cal[n]['K'] for n in names],cam_to_ego=[cal[n]['T_hs_ego_from_opt'] for n in names],ego_pose=f['T_world_from_ego'],timestamp_s=f['timestamp_s'])
            assert synchronized(f,inp)
            ip=dest/(sample.replace('/','_')+'_input.json');write(ip,inp)
            samples.append(dict(sample_id=sample,group=man['group'],split=man['split'],configuration=f['configuration'],map=man['map'],input=str(ip.resolve()),input_sha256=file_hash(ip),raw_frame=str(fp),camera_names=names))
            audits.append(dict(sample_id=sample,pair_transform_delta=delta,synchronized=True))
    write(dest/'index.json',dict(samples=samples));write(dest/'quality.json',dict(kind='offline_actual_CARLA_integrity',records=audits,provenance=provenance()))

if __name__=='__main__':main()
