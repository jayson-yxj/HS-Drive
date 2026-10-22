"""Postrun placement diversity and target-view support; no filtering by accuracy."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m5o_common import DATA,write,provenance


def main():
    records=[]
    for split in ['train','val','calibration','sealed']:
        path=DATA/'processed'/f'm5o_{split}_pixels/index.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())['records'];positions=[];models=Counter();directions=Counter();truncated=Counter()
        for r in rows:
            frame=json.loads(Path(r['raw_frame']).read_text())
            for a in frame['actors']:
                center=np.mean(a['corners_hs_ego_m'],axis=0)
                if a['is_ego'] or np.linalg.norm(center[:2])>80:continue
                positions.append(dict(group=r['group'],frame_id=frame['frame_id'],type_id=a['type_id'],center_ego=center.tolist(),world_xy=np.array(a['world_matrix'])[:2,3].tolist(),world_heading_deg=float(np.degrees(np.arctan2(a['world_matrix'][1][0],a['world_matrix'][0][0])))));models[a['type_id']]+=1
                directions[str(int((np.degrees(np.arctan2(center[1],center[0]))+180)//60)%6)]+=1
        canonical={(x['group'],x['type_id'],*np.round(x['world_xy'],2)) for x in positions}
        records.append(dict(split=split,frames=len(rows),groups=len({r['group'] for r in rows}),actor_placements=len(positions),unique_world_actor_positions=len(canonical),blueprints=dict(models),bearing_sectors=dict(directions),placements=positions))
    write('reports/m5o_sampling_diversity.json',dict(kind='offline_actual_CARLA_metadata',records=records,provenance=provenance(),limits='Independent positions, not repeated frames, counted. Road-based layout family shared across groups. Continuous position variation does not create fully independent semantic scenes. World heading/size distribution depends on road geometry.'))
if __name__=='__main__':main()
