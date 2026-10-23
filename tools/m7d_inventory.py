"""Read all existing M6A/B traces, do not invent missing measured fields."""
import json
from pathlib import Path
import numpy as np
from tools.m7d_common import *

def main():
 records=[]
 paths=sorted(Path('reports/m6b_trials').glob('*/trace.jsonl'))+sorted(Path('reports').glob('m6a*artifacts/*.jsonl'))
 for p in paths:
  rows=[json.loads(s) for s in p.read_text().splitlines()];fields=set.intersection(*(set(x) for x in rows)) if rows else set();missing=[]
  required=['frame','sim_time','measurement','obstacles','timeline']
  for k in required:
   if k not in fields:missing.append(k)
  if not missing:
   for k in ['position','rotation','velocity','speed','long_speed']:
    if not all(k in r['measurement'] for r in rows):missing.append('measurement.'+k)
  ids=set();gaps=None;dt=None;idbad=0;fullwindows=0;actorrows=0;motion=[]
  if not missing:
   times=np.array([r['sim_time'] for r in rows]);frames=np.array([r['frame'] for r in rows]);dt=[float(np.diff(times).min()),float(np.diff(times).max())];gaps=int(np.sum(np.diff(frames)!=1));previous={}
   for r in rows:
    current=set()
    for o in r['obstacles']:
     needed={'actor_id','xy','yaw','half_length','half_width','velocity','frame','timestamp'}
     if not needed<=set(o):missing.append('obstacle_fields');continue
     ids.add(o['actor_id']);current.add(o['actor_id']);actorrows+=1;motion.append(float(np.linalg.norm(o['velocity'])))
     if o['frame']!=r['frame'] or abs(o['timestamp']-r['sim_time'])>1e-6:idbad+=1
     old=previous.get(o['actor_id'])
     if old and abs(r['sim_time']-old[0]-.05)<1e-5 and np.linalg.norm(np.array(o['xy'])-old[1])>2:idbad+=1
     previous[o['actor_id']]=(r['sim_time'],np.array(o['xy']))
    if len(current)!=len(r['obstacles']):idbad+=1
   fullwindows=int(np.sum(times<=times[-1]-6))
  records.append(dict(path=str(p),asset=asset(p),rows=len(rows),missing=sorted(set(missing)),frame_gaps=gaps,dt_range_s=dt,actor_ids=sorted(ids),actor_observations=actorrows,association_or_stamp_issues=idbad,complete_time_windows=fullwindows,support='HistoricalReplayCandidate' if not missing and not gaps and not idbad else 'UnsupportedWithoutFabrication',actor_speed_range=[min(motion),max(motion)] if motion else None,never_new_blind_test=True))
 write('reports/m7d_historical_inventory.json',dict(kind='ActualServerHistoricalTraceFieldAudit',provenance=provenance(),records=records,summary=dict(files=len(records),supported=sum(r['support']=='HistoricalReplayCandidate' for r in records)),limitations='M6B obstacle catalog may exclude unrecorded static world or out-of-scope actors; missing future IDs remain Unknown. M6A fields not filled using interpolation. Historical ego used original privileged control, no M7D execution.'))
 print('audited',len(records),'supported',sum(r['support']=='HistoricalReplayCandidate' for r in records))
if __name__=='__main__':main()
