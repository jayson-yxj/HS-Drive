"""Post-freeze independent map checks. Never modifies candidates or decisions."""
import math
from pathlib import Path
import numpy as np
import carla
from simulator.m6b_bypass import body_samples
from simulator.motion_oracle import wrap
from diagnostics.m7b_motion import interpolate
from diagnostics.m7a_planning import center
from tools.m7b_common import *


def main():
    cfg=read('configs/m7b/protocol.json');m=carla.Map('Town03',Path(cfg['planning_map']).read_text());data=read('reports/m7b_fresh_final_frozen.json');roadmap={x['road']:x for x in data['roads']};placement=[];accepted=[]
    def query(x,y):return m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=0.),project_to_road=False,lane_type=carla.LaneType.Driving)
    for r in data['records']:
        if r['fixture']['status']!='FixtureValid':continue
        f=roadmap[r['road']];scene=r['request']['scene']
        for b in scene['boxes']:
            spec=next(x for x in r['fixture']['placements'] if x['object_id']==b['object_id'])
            if not spec['road_required']:continue
            wp=query(*b['xy']);yaw=-math.radians(wp.transform.rotation.yaw) if wp else None
            error=abs(float(wrap(b['yaw']-yaw))) if wp else None
            placement.append(dict(scene_id=r['scene_id'],object_id=b['object_id'],map_heading_error_rad=error,correct_road_lane=bool(wp and wp.road_id==r['road'] and wp.lane_id==spec['expected_lane']),heading_pass=error is not None and error<=.08))
        for e in r['candidate_evaluations']:
            if e['name'].startswith('E_stop') or not e['robust_accepted']:continue
            c=e['candidate'];q=np.arange(0,6.0001,.05);p=interpolate(c['points'],c['times_s'],q);allowed={f['lane']}
            side='left' if e['name']=='B_left' else 'right' if e['name']=='C_right' else None
            if side in f['sides'] and f['sides'][side]['permitted']:allowed.add(f['sides'][side]['lane'])
            bad=0;tested=0
            for pose in p:
                xy=center(pose,scene['body_center_offset'])
                for xx,yy in body_samples(*xy,pose[2],*(np.array(scene['body_half_extent'])+[.2,.15]),spacing=.4):
                    wp=query(xx,yy);tested+=1
                    if wp is None or wp.road_id!=r['road'] or wp.lane_id not in allowed:bad+=1
            accepted.append(dict(scene_id=r['scene_id'],candidate=e['name'],map_grid_bad=bad,map_grid_tested=tested,stop_reference_gap_ok=e['name']!='D_stop' or e['reference']['min_OBB_gap_m'] is None or e['reference']['min_OBB_gap_m']>=3.,reference_collisions=e['reference']['collisions']))
    write('reports/m7b_independent_map_audit.json',dict(kind='PostEvaluationIndependentSampledCARLAWaypointAudit',provenance=provenance(),placements=placement,accepted=accepted,heading_failures=sum(not r['heading_pass'] or not r['correct_road_lane'] for r in placement),accepted_road_failures=sum(r['map_grid_bad']>0 for r in accepted),stop_reference_gap_failures=sum(not r['stop_reference_gap_ok'] for r in accepted),dt_s=.05,grid_spacing_m=.4,limits='Independent numerical method but same OpenDRIVE, not physical curb or mesh truth; no final retuning'))
    print('independent map audit',len(placement),'placements',len(accepted),'accepted candidates')
if __name__=='__main__':main()
