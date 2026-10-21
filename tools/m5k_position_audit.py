"""Read-only cross-check against historical processed indices, beyond config poses."""
import json
from pathlib import Path
import numpy as np
from tools.m5k_common import write,asset,ROOT
from simulator.recorder import file_hash

def main():
 historic={};indices=[];unread=[]
 for root in ROOT.glob('m5*'):
  if root.name.startswith('m5k'):continue
  p=root/'index.json'
  if not p.exists():continue
  try:d=json.loads(p.read_text())
  except Exception:continue
  rr=d.get('records',d.get('samples',[]))
  if not isinstance(rr,list):continue
  used=False
  for r in rr:
   if not isinstance(r,dict)or 'raw_frame'not in r:continue
   raw=Path(r['raw_frame']);group=r.get('group',str(raw.parent.parent));key=(group,str(raw))
   if key in historic:continue
   if not raw.exists():unread.append(str(raw));continue
   frame=json.loads(raw.read_text());manpath=raw.parent.parent/'manifest.json'
   if not manpath.exists():continue
   man=json.loads(manpath.read_text());T=np.array(frame['T_world_from_ego']);historic[key]=dict(group=group,map=man['map'].split('/')[-1],xy=(T[:2,3]*[1,-1]).tolist());used=True
  if used:indices.append(asset(p))
 new=json.loads(Path('reports/m5k_position_allocation.json').read_text())['groups'];out=[]
 for g in new:
  rr=[r for r in historic.values()if r['map']==g['map']];dist=[float(np.linalg.norm(np.array(r['xy'])-g['pose'][:2]))for r in rr];j=int(np.argmin(dist))if dist else None;out.append(dict(group=g['group'],nearest_processed_group=rr[j]['group']if j is not None else None,min_processed_m=dist[j]if j is not None else None,passed=j is not None and dist[j]>=45))
 write('reports/m5k_split_audit.json',dict(kind='offline_historical_metadata_no_testRGB',historical_frame_positions=len(historic),historical_groups=len({r['group']for r in historic.values()}),indices=indices,new_positions=out,missing_raw_paths=sorted(set(unread)),all_passed=all(r['passed']for r in out),source_sha256=file_hash(__file__),limits='Additionalprocessedindexaudit. M4 drivingallpositions covered by allocator. Pose-distance separation doesnot imply independenttextures,vehicles or weather.'))
if __name__=='__main__':main()
