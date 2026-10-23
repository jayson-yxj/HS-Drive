"""Post-final full-body numerical map/stop audit, no selection changes."""
import math
import numpy as np
import carla
from pathlib import Path
from tools.m7c_common import *
from diagnostics.m7b_motion import interpolate
from diagnostics.m7a_planning import center
from simulator.m6b_bypass import body_samples
from simulator.motion_oracle import wrap


def main():
 m=carla.Map('Town03',Path('reports/m6b_execution_logs/Town03.xodr').read_text());d=read('reports/m7c_fresh_final_frozen.json');roads={r['road']:r for r in d['roads']};out=[];actors=[]
 for r in d['records']:
  if r['fixture']['status']!='FixtureValid':continue
  f=roads[r['road']];scene=r['request']['scene']
  def wp(x,y):return m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=0.),project_to_road=False,lane_type=carla.LaneType.Driving)
  for b in scene['boxes']:
   placement=next(x for x in r['fixture']['placements'] if x['object_id']==b['object_id'])
   if not placement['road_required']:continue
   w=wp(*b['xy']);ok=w is not None and w.road_id==r['road'] and w.lane_id==placement['expected_lane'];error=abs(float(wrap(b['yaw']+math.radians(w.transform.rotation.yaw)))) if w else None
   actors.append(dict(scene=r['scene_id'],actor=b['object_id'],road_lane_ok=ok,heading_error_rad=error,passed=ok and error<.08))
  for e in r['evaluations']:
   if not e['accepted']:continue
   c=e['candidate'];times=np.arange(0,6.00001,.05);p=interpolate(c['points'],c['times_s'],times);allowed={f['lane']};side='left' if e['name']=='B_left' else 'right' if e['name']=='C_right' else None
   if side in f['sides'] and f['sides'][side]['permitted']:allowed.add(f['sides'][side]['lane'])
   bad=0;count=0
   for pose in p:
    for xx,yy in body_samples(*center(pose,scene['body_center_offset']),pose[2],*(np.array(scene['body_half_extent'])+[.2,.15]),spacing=.4):
     w=wp(xx,yy);count+=1
     if w is None or w.road_id!=r['road'] or w.lane_id not in allowed:bad+=1
   stopgap=e['reference']['min_OBB_gap_m'];stopok=e['name']!='D_stop' or stopgap is None or stopgap>=3
   out.append(dict(scene=r['scene_id'],candidate=e['name'],selected=r['selected']==e['name'],body_grid_samples=count,body_grid_violations=bad,stop_reference_gap_ok=stopok,reference_conflicts=e['reference']['collisions'],slow_reserve=e['stop_reserve']))
 write('reports/m7c_independent_final_audit.json',dict(kind='PostFinalIndependentSampledSDKBodyAudit',provenance=provenance(),actor_checks=actors,candidates=out,actor_failures=sum(not x['passed'] for x in actors),accepted_road_failures=sum(x['body_grid_violations']>0 for x in out),stop_gap_failures=sum(not x['stop_reference_gap_ok'] for x in out),future_conflicts=sum(bool(x['reference_conflicts']) for x in out),scope='Different numerical method on same OpenDRIVE.0.05s/0.4m body grid; future collision0.025s Shapely in source; not continuous mesh certificate.'))
 print('actors',len(actors),'accepted',len(out),'road failures',sum(x['body_grid_violations']>0 for x in out))
if __name__=='__main__':main()
