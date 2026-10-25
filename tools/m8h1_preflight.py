"""Single frozen +3m lane displacement per target; no alternative candidate search."""
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.m8h1_common import *
from tools.m8g_history import preflight
from simulator.m8f_placement import validate_request,box_record,pose
from datasets.m8h0_geometry import make_box,footprint,pair_relation

def tf(p):
 import carla
 return carla.Transform(carla.Location(*p['xyz']),carla.Rotation(pitch=p['rotation'][0],yaw=p['rotation'][1],roll=p['rotation'][2]))

def request_at(world,mp,transform,where,spec,contract,env):
 import carla
 # Match exact preserved XY/yaw, using declared map layer only for ground selection.
 t=carla.Transform(carla.Location(transform.location.x,transform.location.y,where.transform.location.z),transform.rotation)
 return validate_request(world,mp,SimpleNamespace(transform=t),spec,contract,env)

def main():
 import carla
 client=carla.Client('127.0.0.1',2100);client.set_timeout(60);w=client.get_world()
 if not w.get_map().name.endswith('Town02'):w=client.load_world('Town02')
 assert client.get_server_version()==client.get_client_version()=='0.9.15';mp=w.get_map();identity=json.loads(Path('reports/m8h0_map_identity_and_runtime.json').read_text());xodr=mp.to_opendrive();assert hashlib.sha256(xodr.encode()).hexdigest()==identity['API_OpenDRIVE']['sha256']
 with (ROOT/'Town02.xodr').open('x') as f:f.write(xodr)
 env=[dict(source='EnvironmentObject',entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],**box_record(e.bounding_box)) for e in w.get_environment_objects(carla.CityObjectLabel.Any)]
 write(ROOT/'environment_inventory.json',dict(map=mp.name,world_episode=w.id,objects=env,frame=w.get_snapshot().frame,source='ActualCARLAOBB_NotMesh'))
 oldenv=json.loads(Path('/root/gpufree-data/hs-drive-experiments/m8h0-native-source-review/environment_inventory.json').read_text())['objects']
 canon=lambda rs:sorted([(r['entity_id'],r['name'],r['semantic_tags'],r['corners_world_m']) for r in rs],key=lambda x:x[0]);assert canon(env)==canon(oldenv),'Static inventory differs from approved H0'
 history=json.loads(HISTORY.read_text());contract=json.loads(Path('configs/m8f/contract.json').read_text());groups=[]
 for t in json.loads(Path('reports/m8h1_frozen_templates.json').read_text())['templates']:
  poses={};reasons=[]
  for role in ['ego','vehicle','person']:
   r=t[role];where=r['waypoint'];wp=mp.get_waypoint_xodr(where['road_id'],where['lane_id'],where['s']);original=tf(r['transform']);present=request_at(w,mp,original,wp,r['asset'],contract,env)
   phases={'present':dict(source_id=r['id'],native_transform=r['transform'],declared_waypoint=where,preflight=present,spawn_transform=r['transform'] if role!='person' else present['request'],spawn_height_basis='Native unchanged' if role!='person' else 'All15SidewalkHeights + frozen localBBoxBottom +0.03m')}
   if role!='ego':
    candidates=wp.next(3.0)
    if len(candidates)!=1 or (candidates[0].road_id,candidates[0].lane_id,candidates[0].section_id)!=(wp.road_id,wp.lane_id,wp.section_id):reasons.append(role+'_ShiftedSourceNotUniqueOrChangedLane')
    else:
     q=candidates[0];oldp=wp.transform;newp=q.transform;shift=carla.Transform(carla.Location(original.location.x+newp.location.x-oldp.location.x,original.location.y+newp.location.y-oldp.location.y,original.location.z+newp.location.z-oldp.location.z),carla.Rotation(pitch=original.rotation.pitch+newp.rotation.pitch-oldp.rotation.pitch,yaw=original.rotation.yaw+newp.rotation.yaw-oldp.rotation.yaw,roll=original.rotation.roll+newp.rotation.roll-oldp.rotation.roll))
     result=request_at(w,mp,shift,q,r['asset'],contract,env)
     phases['shifted']=dict(source_id=r['id']+'__declared_next3m',source_rule='Exactly1waypoint.next(3m), same road/lane/section, preserve source lateral offset',native_transform=pose(shift),declared_waypoint=dict(road_id=q.road_id,lane_id=q.lane_id,section_id=q.section_id,s=q.s,transform=pose(q.transform)),preflight=result,spawn_transform=pose(shift) if role=='vehicle' else result['request'],spawn_height_basis='Native above-road offset preserved' if role=='vehicle' else 'All15SidewalkHeights + frozen localBBoxBottom +0.03m')
   for phase,p in phases.items():
    if p['preflight']['status']!='GroundedByGeometryProxy':reasons.extend(role+'_'+phase+'_'+v for v in p['preflight']['reasons'])
    if p['preflight']['request']:
     p['hypothetical_box']=make_box(p['preflight']['request']['matrix'],r['asset']);p['history']=preflight(footprint(p['hypothetical_box']),'Town02',history)
   poses[role]=dict(asset=r['asset'],phases=phases)
  pairs={}
  for phase in ['present','shifted']:
   members=[('ego',poses['ego']['phases']['present'])]+[(role,poses[role]['phases'].get(phase,{})) for role in ['vehicle','person']]
   pairs[phase]=[]
   for i,(ka,a) in enumerate(members):
    for kb,b in members[i+1:]:
     if 'hypothetical_box' not in a or 'hypothetical_box' not in b:reasons.append('MissingBoxForPair');continue
     relation=pair_relation(a['hypothetical_box'],b['hypothetical_box']);pairs[phase].append(dict(first=ka,second=kb,**relation))
     if not relation['pair_clearance_pass']:reasons.append('ActorEnvelopeConflict_'+phase+'_'+ka+'_'+kb)
  groups.append(dict(template=t['id'],map='Town02',poses=poses,pairs=pairs,status='ShiftedContractGo' if not reasons else 'ShiftedContractNoGo',reasons=sorted(set(reasons)),source=asset('reports/m8h1_frozen_templates.json'),FinalEligible=False))
  print(t['id'],groups[-1]['status'],groups[-1]['reasons'],flush=True)
 write('reports/m8h1_shifted_contract.json',dict(**provenance('ActualCARLAReadOnlyPreflightBeforeActors'),groups=groups,world_episode=w.id,server=client.get_server_version(),map_xodr=asset(ROOT/'Town02.xodr'),environment=asset(ROOT/'environment_inventory.json'),H0_static_inventory_matches=True,no_other_candidate_evaluated=True,per_target_shift_candidates=1))
if __name__=='__main__':main()
