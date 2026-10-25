"""Unified versioned historical index before any candidate allocation."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8g_common import *
from datasets.m8f_quality import polygon_distance
OLD=Path('/root/gpufree-data/hs-drive-experiments/m8f-grounded-quality')
def preflight(poly,town,index,ego_xy=None):
 poly=np.asarray(poly,float);center=poly.mean(0);radius=float(np.linalg.norm(poly-center,axis=1).max());records=[r for r in index['records'] if r['map']==town]
 out={}
 for category,subset in [('actor',[r for r in records if r['target']]),('scene',[r for r in records if not r['target']])]:
  nearest=None
  # Circle lower bound for pruning only; exact polygon distance for recovered OBBs.
  ordered=sorted(subset,key=lambda r:max(0,float(np.linalg.norm(center-r['center_xy']))-radius-r['radius_m']))
  for r in ordered:
   lower=max(0,float(np.linalg.norm(center-r['center_xy']))-radius-r['radius_m'])
   if nearest and lower>nearest['distance_m']:break
   d=polygon_distance(poly,r['polygon']) if r['polygon'] is not None else lower
   if nearest is None or d<nearest['distance_m']:nearest=dict(distance_m=d,method='PolygonXYDistance' if r['polygon'] is not None else 'ConservativeCircleLowerBound',source=r['source'],kind=r['kind'],source_index=r['index_origin'],detail=r.get('detail'))
  out[category]=nearest
 out['ego_distance_to_known_ego_m']=min([float(np.linalg.norm(np.array(ego_xy)-r['center_xy'])) for r in records if not r['target']] or [None]) if ego_xy is not None else None
 out.update(coverage='HistoricalCoverageIncomplete',layout='RepeatedLayoutFamily',map='HistoricalTownReuse',FinalEligible=False,development_use='DiagnosticOnly')
 out['status']='HistoricalTargetOverlap' if out['actor'] and out['actor']['distance_m']<22 else 'HistoricalCoverageIncomplete'
 out['evaluated_scene_overlap']=bool(out['scene'] and out['scene']['distance_m']==0)
 return out

def main():
 records=[];sources={};origins=[]
 for name in ['historical_index.json','historical_raw_frame_supplement.json']:
  p=OLD/name;d=json.loads(p.read_text());origins.append(asset(p))
  for a in d['sources']:
   if sha(a['path'])!=a['sha256']:raise ValueError('Historical source mutated: '+a['path'])
   sources[a['path']]=a
  for r in d['records']:
   q=dict(r);q['radius_m']=q.pop('radius',q.get('radius_m',0));q['target']='Actor' in q['kind'] or 'OBB' in q['kind'];q['index_origin']=name
   q['coordinate']='CARLA_world_x_y_right_z_up';q['map_version']='HistoricalCARLA0.9.15_or_0.9.16_NotEquivalentMesh';q['source_sha256']=sources[q['source']]['sha256'] if q['source'] in sources else sha(q['source'])
   records.append(q)
 index=dict(**provenance('UnifiedHistoricalIndexBeforeAllocation'),records=records,historical_sources=list(sources.values()),origins=origins,coverage='HistoricalCoverageIncomplete',same_road_and_layout_unknown='Not a fresh map or blind final')
 write(ROOT/'unified_history_v1.json',index)
 regs=[]
 old=json.loads(Path('reports/m8f_history_recovery_supplement.json').read_text())['checks']
 req=json.loads(Path('reports/m8f_ray_hit_and_contact_audit.json').read_text())['requests']
 for check in old:
  r=next(r for r in req if r['group']==check['group'] and r['key']==check['target']);p=np.asarray(r['preflight']['present']['requested_obb']['corners_world_m'])[[0,4,6,2],:2]
  raw=dict(index,records=[x for x in records if x['index_origin']=='historical_raw_frame_supplement.json'])
  out=preflight(p,'Town02',raw)
  assert abs(out['actor']['distance_m']-check['nearest_raw_actor']['distance_m'])<1e-6
  regs.append(dict(group=check['group'],target=check['target'],original=check['nearest_raw_actor'],actual=out,passed=True))
 write('reports/m8g_historical_preflight_index.json',dict(**provenance('UnifiedIndexAndKnownOverlapRegression'),index=asset(ROOT/'unified_history_v1.json'),records=len(records),raw_frame_sources=6190,historical_source_count=len(sources),regression=regs,coverage='HistoricalCoverageIncomplete',constructed_before_scan=True,new_Final=0))
 print(len(records),'records',len(sources),'sources; all5footprint regressions exact',flush=True)
if __name__=='__main__':main()
