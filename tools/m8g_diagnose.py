"""Replay all frozen requests and attribute every rejected point conservatively."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from tools.m8g_common import *
from tools.m8g_history import OLD
from datasets.m8g_geometry import planned_box,inside_box,overhead_class,OpenDriveGeometry
from simulator.m8f_placement import validate_request,box_record,pose,xyz

class RenderEvidence:
 def __init__(self, group):
  self.path=OLD/'capture'/group/'evaluator/empty_sidecar.json';self.side=json.loads(self.path.read_text());self.cams=[]
  for c in self.side['calibration']:
   s=self.side['sensors'][c['name']]
   self.cams.append((c,np.load(s['depth']['path'])['depth_m'],np.load(s['semantic']['path'])['semantic'],np.load(s['instance']['path'])['instance']))
 def hit(self,point):
  out=[]
  for c,d,sem,ins in self.cams:
   p=np.linalg.solve(np.asarray(c['T_world_from_camera']),np.r_[point,1])[:3];opt=np.array([p[1],-p[2],p[0]])
   if opt[2]<=0:continue
   uv=np.asarray(c['K'])@opt;u,v=np.rint(uv[:2]/uv[2]).astype(int)
   if not (0<=u<d.shape[1] and 0<=v<d.shape[0]):continue
   out.append(dict(camera=c['name'],pixel=[int(u),int(v)],projected_depth_m=float(opt[2]),render_depth_m=float(d[v,u]),depth_residual_m=float(d[v,u]-opt[2]),semantic=int(sem[v,u]),instance_code=int(ins[v,u]),agrees=bool(abs(d[v,u]-opt[2])<=.3),reference='HistoricalM8F_Renderer_NotIndependentMesh',frame_id=self.side['frame_id']))
  return out
 def points_in_box(self,box):
  hits=[]
  for c,d,sem,ins in self.cams:
   # Full image, no decimation: absence still not proof of empty volume.
   yy,xx=np.indices(d.shape);k=np.asarray(c['K']);z=d.reshape(-1);p=np.c_[(xx.reshape(-1)-k[0,2])*z/k[0,0],(yy.reshape(-1)-k[1,2])*z/k[1,1],z]
   carla=np.c_[p[:,2],p[:,0],-p[:,1]];m=np.asarray(c['T_world_from_camera']);world=carla@m[:3,:3].T+m[:3,3];bm=np.linalg.inv(box['T_world_from_box']);local=world@bm[:3,:3].T+bm[:3,3]
   good=np.all(np.abs(local)<np.asarray(box['extent_m'])-.03,axis=1)&(z>0)&(z<100)
   ids=np.flatnonzero(good);hits.append(dict(camera=c['name'],count=len(ids),semantic_counts=dict(Counter(map(str,sem.reshape(-1)[ids]))),sample=[dict(xyz=world[j].tolist(),pixel=[int(j%d.shape[1]),int(j//d.shape[1])],semantic=int(sem.reshape(-1)[j])) for j in ids[::max(1,len(ids)//25)][:25]]))
  return hits

def main():
 import carla
 ROOT.mkdir(exist_ok=True);client=carla.Client('127.0.0.1',2100);client.set_timeout(45);w=client.get_world();mp=w.get_map();assert mp.name.endswith('Town02')
 inv=[dict(entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],**box_record(e.bounding_box)) for e in w.get_environment_objects(carla.CityObjectLabel.Any)]
 actors=[dict(entity_id=a.id,name=a.type_id,semantic_tags=list(a.semantic_tags),**box_record(a.bounding_box,a.get_transform())) for a in w.get_actors() if not a.type_id.startswith(('sensor.','spectator','controller.'))]
 write(ROOT/'live_inventory.json',dict(world_episode=w.id,frame=w.get_snapshot().frame,server=client.get_server_version(),map=mp.name,xodr_sha256=sha(ROOT/'Town02.xodr'),environment=inv,actors=actors))
 od=OpenDriveGeometry(mp.to_opendrive());contract=json.loads(Path('configs/m8f/contract.json').read_text());protocol=json.loads(Path('configs/m8f/protocol.json').read_text());frozen=json.loads(Path('reports/m8f_ray_hit_and_contact_audit.json').read_text())['requests']
 freeze_files=['reports/m8f_summary.md','reports/m8f_acceptance.json','reports/m8f_ray_hit_and_contact_audit.json','reports/m8f_natural_scene_quality.json','reports/m8f_split_preflight.json','reports/m8f_history_recovery_supplement.json','simulator/m8f_placement.py','datasets/m8f_quality.py','tools/m8f_allocate.py','configs/m8f/contract.json','configs/m8f/protocol.json']
 write(ROOT/'frozen_m8f_inputs.json',dict(**provenance('ReadOnlyM8FFreeze'),assets=[asset(p) for p in freeze_files],requests=frozen))
 all_hits=[];lanes=[];statics=[];rows=[];replay=[]
 for group in protocol['groups']:
  render=RenderEvidence(group['group']);oldinv=json.loads(Path(render.side['environment_inventory']['path']).read_text())['objects'];sources=oldinv+render.side['objects']
  for spec in group['objects']:
   orig=next(x for x in frozen if x['group']==group['group'] and x['key']==spec['key']);classes=set();phase_rows=[]
   for phase,where in spec['phases'].items():
    old=orig['preflight'][phase];wp=mp.get_waypoint_xodr(where['road_id'],where['lane_id'],where['s']);new=validate_request(w,mp,wp,spec['asset'],contract,inv,where['offset_m']);box=planned_box(old)
    ray_equal=all(len(a['hits'])==len(b['hits']) and all(x['semantic']==y['semantic'] and np.linalg.norm(np.array(x['xyz'])-y['xyz'])<.01 for x,y in zip(a['hits'],b['hits'])) for a,b in zip(old['rays'],new['rays']))
    replay.append(dict(group=group['group'],key=spec['key'],phase=phase,A=new,B=dict(placement=new['status'],reasons=new['reasons'],diagnostic_only=True),frozen_status=old['status'],status_equal=old['status']==new['status'],reasons_equal=old['reasons']==new['reasons'],all_ray_hits_equal_within_1cm=ray_equal,all_lane_decisions_equal=[r['ok'] for r in old['lane_checks']]==[r['ok'] for r in new['lane_checks']],selected_ground_equal=all(a['decision']==b['decision'] for a,b in zip(old['rays'],new['rays'])),static_competitors_equal=old['static_competitors']==new['static_competitors'],planned_obb_equal=old['requested_obb']==new['requested_obb'],support_gap_policy='Unchanged; actual placement not attempted in this replay',unknown_override=False))
    if box is None:classes.add('MissingGroundEvidence')
    sample_lanes=[]
    for i,(r,ln) in enumerate(zip(old['rays'],old['lane_checks'])):
     point=ln['point'];near=mp.get_waypoint(carla.Location(*point),project_to_road=True,lane_type=carla.LaneType.Any)
     try:
      geo=od.point(where['road_id'],where['lane_id'],where['s'],point)
      closure=float(np.linalg.norm(od.lane_center(where['road_id'],where['lane_id'],where['s'])[:2]-np.array(xyz(wp.transform.location)[:2])))
      ground=r['decision'].get('selected');support=ground is not None and abs(ground['xyz'][2]-point[2])<=.5
      if closure>.02:status='InsufficientRoadGeometry'
      elif not geo['within']:status='ActuallyOutsideDeclaredLane'
      elif not support:status='InsufficientRoadGeometry'
      elif not ln['ok']:status='OpenDRIVEBoundaryDisagreement'
      else:status='ValidWithinConservativeLaneProof'
     except (ValueError,KeyError) as ex:geo=dict(error=str(ex));closure=None;status='InsufficientRoadGeometry'
     if status=='ActuallyOutsideDeclaredLane':classes.add('VerifiedInvalidPlacement')
     elif status in ['OpenDRIVEBoundaryDisagreement','InsufficientRoadGeometry']:classes.add('MapQueryAmbiguity')
     item=dict(group=group['group'],key=spec['key'],phase=phase,sample=i,is_bottom_corner=i in [0,2,12,14],original_point=point,declared=where,original=ln,fixed_road_lane_s=pose(wp.transform),nearest_diagnostic_only=None if near is None else dict(road_id=near.road_id,lane_id=near.lane_id,section_id=near.section_id,s=near.s,width=near.lane_width,pose=pose(near.transform)),independent=geo,center_closure_m=closure,ground_hits=r['hits'],status=status)
     lanes.append(item);sample_lanes.append(status)
     for h in r['hits']:
      selected=r['decision'].get('selected')
      if selected is None or h['semantic'] in old['declared_region']['semantics'] or h['xyz'][2]<=selected['xyz'][2]+.1:continue
      competitors=[o for o in sources if inside_box(h['xyz'],o)]
      # Road/ground OBBs covering a region are not a semantic source for NONE or vegetation.
      plausible=[o for o in competitors if int(o['semantic_tags'][0] if o['semantic_tags'] else -1) not in [1,2,24]]
      renders=render.hit(h['xyz']);agrees=any(a['agrees'] and (h['semantic']==0 or a['semantic']==h['semantic']) for a in renders)
      status=overhead_class(h,box,plausible,agrees) if box else 'UnidentifiedRayHit'
      if status=='SourceQualifiedCollision':classes.add('VerifiedInvalidPlacement')
      elif status=='VerifiedNonIntersectingOverhead':classes.add('ConservativeRejectionUnresolved')
      elif status in ['UnidentifiedRayHit','ConflictingGeometry']:classes.add('UnknownCollisionSource')
      all_hits.append(dict(group=group['group'],key=spec['key'],phase=phase,sample=i,all_ray_hits=r,hit=h,height_above_ground_m=h['xyz'][2]-selected['xyz'][2],hypothetical_box=box,hit_inside_planned_volume=False if box is None else inside_box(h['xyz'],box,0),clearance_to_top_m=None if box is None else h['xyz'][2]-box['top_z'],competing_sources=competitors,plausible_sources=plausible,render_evidence=renders,multiple_ground_levels=r['decision'].get('ground_levels'),status=status,tree_category=('A' if status=='SourceQualifiedCollision' else 'B' if status=='VerifiedNonIntersectingOverhead' else 'C') if h['semantic']==9 else None))
    for comp in old['static_competitors']:
     source=next(o for o in oldinv if o['entity_id']==comp['entity_id']);surface=render.points_in_box(box)
     # Render surface intrusion proves occupied target space, not the identity of a mesh.
     relevant=[p for s in surface for p in s['sample'] if inside_box(p['xyz'],source,0) and p['semantic'] not in [0,1,2,24]]
     status='ConfirmedGeometryConflict' if relevant else 'ConservativeOBBConflict'
     if relevant:classes.add('VerifiedInvalidPlacement')
     else:classes.add('ConservativeRejectionUnresolved')
     a=np.array(box['corners_world_m']);b=np.array(source['corners_world_m']);statics.append(dict(group=group['group'],key=spec['key'],phase=phase,competitor=source,target=box,z_overlap_m=float(max(0,min(a[:,2].max(),b[:,2].max())-max(a[:,2].min(),b[:,2].min()))),xy_geometry='Frozen polygon overlap; conservative OBB not exact mesh',render_intrusion=surface,source_compatible_intrusion_samples=relevant,status=status))
    phase_rows.append(dict(phase=phase,reasons=old['reasons'],lane_counts=dict(Counter(sample_lanes))))
   if orig['accepted_for_spawn']:classes.add('GeometryProxySupported')
   if not classes:classes.add('ConservativeRejectionUnresolved')
   exclusive='AcceptedHistoricalProxy' if orig['accepted_for_spawn'] else 'ConfirmedInvalid' if 'VerifiedInvalidPlacement' in classes else 'ProvenConservativeFalseReject' if classes=={'ConservativeRejectionUnresolved'} and all(x['status']=='VerifiedNonIntersectingOverhead' for x in all_hits if x['group']==group['group'] and x['key']==spec['key']) and not any(x['key']==spec['key'] and x['group']==group['group'] for x in statics) else 'Unresolved'
   rows.append(dict(group=group['group'],key=spec['key'],category=spec['asset']['category'],historical_accepted=orig['accepted_for_spawn'],classification=sorted(classes),exclusive=exclusive,phases=phase_rows))
   print(group['group'],spec['key'],exclusive,sorted(classes),flush=True)
 write(ROOT/'full_replay.json',dict(**provenance('ActualCARLA_FrozenValidatorReplay'),records=replay))
 write('reports/m8g_rejection_attribution.json',dict(**provenance('OfflineHistoricalEvidenceAndActualCARLAReplay'),requests=rows,exclusive_counts=dict(Counter(r['exclusive'] for r in rows)),multi_label_counts=dict(Counter(k for r in rows for k in r['classification'])),frozen=asset(ROOT/'frozen_m8f_inputs.json'),no_physical_contact_certification=True))
 for name,records in [('none_hit_analysis',[r for r in all_hits if r['hit']['semantic']==0]),('overhead_clearance',[r for r in all_hits if r['hit']['semantic']!=0]),('lane_boundary_differential',lanes),('static_geometry_conflicts',statics)]:
  write('reports/m8g_'+name+'.json',dict(**provenance('ActualCARLAQueriesAndHistoricalRenderedEvidence'),records=records,status_counts=dict(Counter(r['status'] for r in records)),map_xodr=asset(ROOT/'Town02.xodr'),old_renderer_reference='Same CARLA renderer, different modalities; not independent mesh truth',acceptance_override=False))
 write('reports/m8g_validator_differential.json',dict(**provenance('ActualCARLA_A_Frozen_B_AnnotatedNoOverride'),replay=asset(ROOT/'full_replay.json'),phase_count=len(replay),A_B_placement_identical=True,newly_accepted=[],frozen_reproduction={k:sum(bool(r[k]) for r in replay) for k in ['status_equal','reasons_equal','all_ray_hits_equal_within_1cm','all_lane_decisions_equal','selected_ground_equal','static_competitors_equal','planned_obb_equal']},history_required_before_allocation=True,diagnostic_never_approves_rejected_targets=True))
if __name__=='__main__':main()
