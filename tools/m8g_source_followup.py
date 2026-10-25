"""Read-only follow-up at the exact audited rays, not additional placement search."""
import json,urllib.request
from pathlib import Path
import numpy as np
from tools.m8g_common import *
from simulator.m8f_placement import box_record,ray
from datasets.m8g_geometry import inside_box

def main():
 import carla
 sourcebase='https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/'
 paths=['LibCarla/source/carla/road/Map.cpp','Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/RayTracer.cpp','Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/BoundingBoxCalculator.cpp']
 official=[];(ROOT/'official_source').mkdir(exist_ok=True)
 for p in paths:
  out=ROOT/'official_source'/Path(p).name
  try:
   if not out.exists():
    with urllib.request.urlopen(sourcebase+p,timeout=10) as response:content=response.read()
    with out.open('xb') as f:f.write(content)
   official.append(dict(url=sourcebase+p,asset=asset(out)))
  except (TimeoutError,OSError) as ex:
   official.append(dict(url=sourcebase+p,local_download='Unavailable',error=type(ex).__name__,review='Official source viewed with web tool; no local hash claim'))
 c=carla.Client('127.0.0.1',2100);c.set_timeout(60);w=c.load_world('Town02');rows=[]
 actors=list(w.get_actors());traffic=[]
 for a in actors:
  if a.type_id.startswith('traffic.') and hasattr(a,'trigger_volume'):
   traffic.append(dict(entity_id=a.id,name=a.type_id,trigger=box_record(a.trigger_volume,a.get_transform()),api_box=box_record(a.bounding_box,a.get_transform())))
 for name in ['none_hit_analysis','overhead_clearance']:
  d=json.loads(Path('reports/m8g_'+name+'.json').read_text())
  for r in d['records']:
   p=r['hit']['xyz'];oldray=r['all_ray_hits'];start=oldray['start'];direction=np.array(oldray['end'])-start;length=float(np.linalg.norm(direction));direction/=length
   pr=w.project_point(carla.Location(*start),carla.Vector3D(*direction),length)
   projected=None if pr is None else dict(xyz=[pr.location.x,pr.location.y,pr.location.z],semantic=int(pr.label),semantic_name=str(pr.label))
   containers=[a for a in traffic if inside_box(p,a['api_box'],.05)]
   matches=[]
   for a in containers:
    b=a['api_box'];local=np.linalg.solve(np.array(b['T_world_from_box']),np.r_[p,1])[:3];matches.append(dict(actor=a,top_face_residual_m=float(abs(local[2]-b['extent_m'][2]))))
   rows.append(dict(group=r['group'],key=r['key'],phase=r['phase'],sample=r['sample'],original_hit=r['hit'],fresh_overlap_channel_ray=ray(w,start,oldray['end']),camera_channel_project_point=projected,traffic_box_candidates=matches,render_agreement=any(a['agrees'] for a in r['render_evidence']),interpretation='TriggerVolumeHypothesisSupported_NotPhysicalCollisionProof' if matches and min(x['top_face_residual_m'] for x in matches)<.01 and projected is not None and projected['semantic']!=0 else 'SourceStillUnresolved',placement_override=False))
 lanes=[]
 for r in json.loads(Path('reports/m8g_lane_boundary_differential.json').read_text())['records']:
  if r['status']!='OpenDRIVEBoundaryDisagreement':continue
  p=carla.Location(*r['original_point']);wp=w.get_map().get_waypoint(p,project_to_road=False,lane_type=carla.LaneType.Driving)
  lanes.append(dict(group=r['group'],key=r['key'],phase=r['phase'],sample=r['sample'],any_original=r['original'],driving_only_query=None if wp is None else dict(road_id=wp.road_id,lane_id=wp.lane_id,section_id=wp.section_id,s=wp.s,width=wp.lane_width),use='DiagnosisOnly_NoAcceptanceOverride',independent_margin_m=r['independent']['margin_m']))
 write('reports/m8g_source_geometry_followup.json',dict(**provenance('ActualCARLAExactFrozenPointFollowup_NoNewCandidates'),official_sources=official,server=c.get_server_version(),world_episode=w.id,ray_rows=rows,lane_rows=lanes,trigger_supported_hits=sum(r['interpretation'].startswith('Trigger') for r in rows),driving_query_returns=sum(r['driving_only_query'] is not None for r in lanes),limitations=['API cast_ray returns location and semantic but no hit component identity','Traffic Actor bounding boxes may represent trigger volumes, not rendered physical mesh','project_point uses a different collision channel; cannot establish full target volume free','No rule changed; these probes are not extra scan candidates']))
 print('trigger-supported',sum(r['interpretation'].startswith('Trigger') for r in rows),'driving query',sum(r['driving_only_query'] is not None for r in lanes),'/',len(lanes),flush=True)
if __name__=='__main__':main()
