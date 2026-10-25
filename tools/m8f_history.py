"""Recover auditable historical regions; missing history is not fresh space."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8f_common import *
from datasets.m8f_quality import polygon,polygon_distance

def main():
 records=[];sources=[];gaps=[];seen=set()
 def add(town,xy,radius,kind,path,detail,poly=None):
  key=(town,round(xy[0],1),round(xy[1],1),round(radius,1),kind)
  if key in seen:return
  seen.add(key);records.append(dict(map=town,center_xy=list(xy),radius_m=radius,kind=kind,source=str(path),detail=detail,polygon=poly))
 for p in sorted(Path('configs').rglob('*.json')):
  if not any(x in str(p) for x in ['m5','m6','m7','m8']):continue
  if 'm8f' in str(p):continue
  try:d=json.loads(p.read_text())
  except ValueError:continue
  sources.append(asset(p))
  if not isinstance(d,dict):continue
  for g in d.get('groups',[]):
   if isinstance(g,dict) and 'pose' in g and 'map' in g:
    add(g['map'].split('/')[-1].replace('_Opt',''),g['pose'][:2],80,'DeclaredSceneAndCameraEnvelope',p,g.get('group','unknown'))
  if any(x in str(p) for x in ['m6','m7']):gaps.append(dict(source=str(p),reason='Declared roads/episodes may lack full camera/target inventory; traces recovered separately'))
 for p in sorted(Path('runs').glob('m5*/supervision.json')):
  d=json.loads(p.read_text());sources.append(asset(p))
  for a in d.get('actors',[]):
   if 'box_center_world_carla' not in a or 'bbox_dimensions_xyz' not in a:continue
   town=a['group'].split('_')[0];xy=a['box_center_world_carla'][:2];rad=float(np.linalg.norm(a['bbox_dimensions_xyz'][:2])/2)
   add(town,xy,rad,'ActorOBBConservativeCircle_OrientationMissing',p,a['group'])
 for p in sorted(Path('reports').glob('m[67]*/**/trace.jsonl')):
  sources.append(asset(p));count=0
  with p.open() as f:
   for line in f:
    d=json.loads(line);pos=d.get('measurement',{}).get('position') or d.get('ego',{}).get('position_carla')
    if pos:add('Town03',pos[:2],80,'ExecutedSceneCameraEnvelope_Town03Protocol',p,str(d.get('frame')));count+=1
    for a in d.get('actors',d.get('obstacles',[])):
     if all(k in a for k in ['xy','yaw','half_length','half_width']):
      xy=[a['xy'][0],-a['xy'][1]];ext=[a['half_length'],a['half_width']]
      add('Town03',xy,float(np.linalg.norm(ext)),'MeasuredActorPlanarOBB',p,str(d.get('frame')),polygon(xy,ext,-a['yaw']).tolist())
  if not count:gaps.append(dict(source=str(p),reason='No recognized world measurement; history incomplete'))
 old=Path('/root/gpufree-data/hs-drive-experiments/m8e-groundtruth')
 for p in sorted((old/'capture').glob('*/evaluator/*_sidecar.json')):
  d=json.loads(p.read_text());sources.append(asset(p));town=d['map'].split('/')[-1]
  for a in d['objects']:
   if not a['is_controlled_target'] and a['entity_id'] not in d['controlled_entity_ids']:continue
   xy=a['center_world_m'][:2]
   if np.linalg.norm(xy)>5000:continue
   corners=np.asarray(a['corners_world_m']);points=corners[[0,4,6,2],:2].tolist()
   add(town,xy,float(np.linalg.norm(a['extent_m'][:2])),'RecordedM8EOBB',p,str(a['entity_id']),points)
 write(ROOT/'historical_index.json',dict(records=records,sources=sources,gaps=gaps,coverage='HistoricalCoverageIncomplete',known_maps=sorted({r['map'] for r in records})))
 # Every original80 object receives explicit attribution, including missing measurements.
 audit=json.loads(Path('reports/m8e_instance_identity_audit.json').read_text())['object_rows'];rows=[]
 for a in audit:
  town=a['map'].split('/')[-1];p=np.asarray(a['corners_world_m'])[[0,4,6,2],:2]
  older=[r for r in records if r['map']==town and r['kind']!='RecordedM8EOBB' and 'Actor' in r['kind']]
  nearest=min([max(0,float(np.linalg.norm(np.array(a['center_world_m'][:2])-r['center_xy']))-float(np.linalg.norm(a['extent_m'][:2]))-r['radius_m']) for r in older] or [1e6])
  floating=(a['group'],a['entity_key']) in [('m8e_development_02','Actor:717'),('m8e_development_02','Actor:715'),('m8e_calibration_00','Actor:1022')]
  rows.append(dict(group=a['group'],entity_key=a['entity_key'],GroundHitUnknown=True,SurfaceTypeUnknown=True,
    OffRoadOrInvalidSurface='SuspectedByPriorVisualReview' if floating else 'Unknown',ApparentFloating=floating,
    ContactNotVerified=True,RenderIdentitySupported=a['identity_grade'].startswith('RenderLinked'),
    HistoryFootprintOverlap=nearest<20,nearest_prior_footprint_lower_bound_m=nearest,
    NaturalSceneEligibilityUnknown=True,original_grade=a['identity_grade'],ray_records='NotRecorded_NoReconstruction',training_eligible=False))
 write('reports/m8f_legacy_failure_attribution.json',dict(**provenance('OfflineReadOnlyM8EAttribution'),targets=80,rows=rows,original_grade_counts=dict(Counter(a['identity_grade'] for a in audit)),original_far_render_targets=21,historical_acceptance_changed=False))
 write('reports/m8f_historical_target_footprints.json',dict(**provenance('OfflineHistoricalSpatialRecovery'),index=asset(ROOT/'historical_index.json'),counts=dict(Counter(r['kind'] for r in records)),maps=sorted({r['map'] for r in records}),coverage='HistoricalCoverageIncomplete',unknown_policy='Cannot certify fresh Final on historical maps with unrecovered evidence',conservatism='80m isotropic scene/camera envelope; orientation-missing boxes use enclosing circles; broad rejection, not exact FOV',layout_policy='RepeatedLayoutFamily across historical CARLA assets; not fresh-map evidence'))
 print(len(records),'spatial records',len(sources),'sources',len(gaps),'coverage gaps',flush=True)
if __name__=='__main__':main()
