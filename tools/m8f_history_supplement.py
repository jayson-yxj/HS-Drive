"""Raw-frame history recovery, preserving the earlier preflight index verbatim."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from geometry.reference import transform_points
from datasets.m8f_quality import polygon_distance
from tools.m8f_common import *

def main():
 rows=[];sources=[];seen=set();skipped=0
 for p in sorted(Path('/root/gpufree-data/hs-drive-data/raw').glob('m5*/*/frames/*.json')):
  d=json.loads(p.read_text());sources.append(asset(p));town=p.parent.parent.name.split('_')[0]
  if 'T_world_from_ego' not in d:skipped+=1;continue
  T=np.asarray(d['T_world_from_ego']);ego=T[:2,3]*[1,-1]
  key=(town,'Ego',*np.round(ego,1))
  if key not in seen:seen.add(key);rows.append(dict(map=town,kind='RawHistoricalEgo',center_xy=ego.tolist(),radius=80,polygon=None,source=str(p)))
  for a in d.get('actors',[]):
   if a.get('is_ego') or 'corners_hs_ego_m' not in a:continue
   corners=np.asarray(a['corners_hs_ego_m'])
   if np.linalg.norm(corners.mean(0)[:2])>90:continue
   world=transform_points(corners,T)*[1,-1,1];center=world.mean(0);key=(town,'Actor',a.get('type_id'),*np.round(center,1))
   if key in seen:continue
   seen.add(key);rows.append(dict(map=town,kind='RawHistoricalActorOBB',center_xy=center[:2].tolist(),radius=float(np.max(np.linalg.norm(world[:,:2]-center[:2],axis=1))),polygon=world[[0,4,6,2],:2].tolist(),source=str(p),actor_id=a.get('actor_id'),frame=d['frame_id']))
 write(ROOT/'historical_raw_frame_supplement.json',dict(records=rows,sources=sources,skipped_frames=skipped,coordinate='RawHSWorld to CARLA via y-negation; corners_hs_ego transformed by recorded T_world_from_ego',coverage='HistoricalCoverageIncomplete'))
 checks=[]
 for g in json.loads(Path('configs/m8f/protocol.json').read_text())['groups']:
  old=[r for r in rows if r['map']==g['map']]
  q=json.loads((ROOT/'capture'/g['group']/'evaluator/grounded_requests.json').read_text())
  for r in q['requests']:
   box=r['preflight']['present'].get('requested_obb')
   if not box:continue
   poly=np.asarray(box['corners_world_m'])[[0,4,6,2],:2];nearest=None
   for h in old:
    if h['kind']!='RawHistoricalActorOBB':continue
    dist=polygon_distance(poly,h['polygon'])
    if nearest is None or dist<nearest['distance_m']:nearest=dict(distance_m=dist,source=h['source'],actor_id=h['actor_id'],frame=h['frame'])
   checks.append(dict(group=g['group'],target=r['key'],nearest_raw_actor=nearest))
 write('reports/m8f_history_recovery_supplement.json',dict(**provenance('OfflineRawHistoricalRecovery_AfterSmoke_NoSelectionChange'),raw_frames=len(sources),unique_records=len(rows),counts=dict(Counter(r['kind'] for r in rows)),asset_index=asset(ROOT/'historical_raw_frame_supplement.json'),checks=checks,preflight_index_unchanged=True,new_final_groups=0,coverage='HistoricalCoverageIncomplete',caveat='Additional source recovery after smoke; cannot retroactively certify preflight or promote old scenes to independent Final. M6/M7 Town03 assumption based on protocol remains conservative, CARLA backend versions differ.'))
 print(len(sources),'raw frames',len(rows),'deduplicated geometric records',flush=True)
if __name__=='__main__':main()
