"""Read-only capture health stream; completed manifests only, never changes selection."""
import json,time
from pathlib import Path
import numpy as np
from tools.m5o_common import DATA,write,provenance
from datasets.m5n_reference import actor_reference


def audit_group(g):
    root=DATA/'raw'/f"m5o_v3_{g['split']}"/g['group']/g['group'];m=json.loads((root/'manifest.json').read_text());cal=json.loads((root/'calibration.json').read_text())['cameras'];records=[];failures=[]
    for fid in m['frames']:
        frame=json.loads((root/'frames'/f'{fid:08d}.json').read_text());ref=json.loads((root/'frames'/f"{m['references'][str(fid)]:08d}.json").read_text());byid={a['actor_id']:a for a in ref['actors']}
        for a in frame['actors']:
            if a['is_ego']:continue
            delta=float(np.max(np.abs(np.array(a['world_matrix'])-byid[a['actor_id']]['world_matrix'])))
            if delta>=.001:failures.append([fid,a['actor_id'],'pair_delta',delta])
            center=np.mean(a['corners_hs_ego_m'],axis=0)
            if np.linalg.norm(center[:2])>80:continue
            closure=actor_reference(a,frame['T_world_from_ego'],np.array(cal['front']['T_hs_ego_from_opt']),np.array(cal['front']['K']))
            records.append(dict(frame_id=fid,actor_id=a['actor_id'],type_id=a['type_id'],center_ego=center.tolist(),road_probe=a['contact_reference']['road_probe_found'],pair_delta=delta,closure=closure['world_closure_residual_m']))
    return dict(group=g['group'],split=g['split'],frames=len(m['frames']),actors=records,failures=failures)


def main():
    gs=json.loads(Path('configs/m5o_capture_dev_v3.json').read_text())['groups'];seen=set()
    while len(seen)<len(gs):
        for g in gs:
            if g['group'] in seen:continue
            path=DATA/'raw'/f"m5o_v3_{g['split']}"/g['group']/g['group']/'manifest.json'
            if not path.exists():continue
            result=audit_group(g);write(f"runs/m5o_logs/geometry_{g['group']}.json",result);seen.add(g['group']);print(g['group'],'frames',result['frames'],'failures',len(result['failures']),flush=True)
        time.sleep(5)
if __name__=='__main__':main()
