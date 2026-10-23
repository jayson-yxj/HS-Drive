"""Topology segment metadata and historical full-body support audit."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import carla
from shapely.geometry import Polygon
from tools.m7c_common import *
from tools.m7c_scenarios import connected_fixture
from tools.m7c_equivalence import request_from
from diagnostics.m7b_motion import ObservedMotion,build_hypotheses,longitudinal
from simulator.oracle_driver import footprint


def main():
 cfg=read('configs/m7b/protocol.json');m=carla.Map('Town03',Path(cfg['planning_map']).read_text());out=[];segments=[]
 for road,filename in [(18,'historical_regression_frozen'),(19,'validation_frozen')]:
  f=connected_fixture(m,road,'historical_support');nodes=f['continuation']['nodes'];groups=[]
  for n in nodes:
   key=(n['road'],n['lane'],n['section'])
   if not groups or groups[-1]['key']!=key:groups.append(dict(key=key,nodes=[]))
   groups[-1]['nodes'].append(n)
  arc=0
  for i,g in enumerate(groups):
   points=np.array([n['pose'] for n in g['nodes']]);length=float(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).sum());segments.append(dict(origin_road=road,road_lane_section=g['key'],sampled_length_m=length,cumulative_segment_start_m=arc,predecessor=groups[i-1]['key'] if i else None,successor=groups[i+1]['key'] if i+1<len(groups) else None,width_range=[min(n['width'] for n in g['nodes']),max(n['width'] for n in g['nodes'])],junction=any(n['junction'] for n in g['nodes']),direction='Waypoint.next_forward',selection='OnlyUniqueLegalSuccessor',stop_reason=f['continuation']['reason']));arc+=length
  for r in read('reports/m7b_'+filename+'.json')['records']:
   if r['road']!=road or r['spec']['kind']!='lead':continue
   req=request_from(r);spec=r['spec'];hs,u=build_hypotheses(req.scene,[ObservedMotion('lead',f['arc'],spec['d'])],cfg);b=req.scene.boxes[0];region=f['polygons'][f['lane']];bad=[]
   for h in hs:
    if any(not region.covers(Polygon(footprint(*p[:2],p[2],b.half_length,b.half_width))) for p in h.poses):bad.append(h.name)
   t=np.arange(0,6.000001,.025);ds,v=longitudinal(t,spec['v'],spec['a'],spec['onset']);ref_support=spec['d']+ds.max()<=f['arc'].length;refroad=None
   if ref_support:refroad=all(region.covers(Polygon(footprint(*p[:2],p[2],b.half_length,b.half_width))) for p in f['arc'].at(spec['d']+ds))
   out.append(dict(scene=r['scene_id'],all_hypotheses_supported=not u and not bad,route_unknown=u,body_road_unsupported_hypotheses=bad,reference_same_parameters_supported=bool(ref_support),reference_body_road_pass=refroad,decisions_reexecuted=False))
 write('reports/m7c_connected_body_support.json',dict(kind='HistoricalPostEvaluationBodySupportAudit',provenance=provenance(),segments=segments,records=out,scope='Actual CARLA topology with sampled OpenDRIVE corridor OBB coverage; same physical reference limitations, no changes to prior outcomes'))
 print('body supported',sum(r['all_hypotheses_supported'] for r in out),'/',len(out))
if __name__=='__main__':main()
