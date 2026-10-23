"""Independent dense Shapely overlap / CARLA point audit; no rule tuning."""
from pathlib import Path
from collections import Counter
import math
import numpy as np
import carla
from shapely.geometry import Polygon
from simulator.oracle_driver import footprint,wrap
from simulator.m6b_bypass import body_samples
from tools.m7a_common import read,write,provenance,asset


def interp(points,times,q):
    a=np.asarray(points);t=np.asarray(times)
    return np.c_[np.interp(q,t,a[:,0]),np.interp(q,t,a[:,1]),np.interp(q,t,np.unwrap(a[:,2]))]


def main():
    cmap=carla.Map('Town03',Path('reports/m6b_execution_logs/Town03.xodr').read_text())
    records=[];locations={};fixture_issues=[]
    for split in ['development_v2','validation','final']:
        d=read('reports/m7a_planner_'+split+'.json')
        for cat in d['catalog']:locations[cat['road']]=cat
        for r in d['records']:
            req=r['request'];scene=req['scene'];half=np.array(scene['body_half_extent'])+[.2,.15];offset=scene['body_center_offset']
            cat=next(c for c in d['catalog'] if c['scene_id']==r['scene_id'])
            obstacles=scene['boxes'];route=np.array(req['route'])
            # Confirm scenario generator's promised "ahead on route" actors.
            for o in obstacles:
                if not o['object_id'].startswith(('block','obstacle')):continue
                xy=np.array(o['xy']);delta=np.diff(route[:,:2],axis=0);l2=(delta**2).sum(-1)
                frac=np.clip(((xy-route[:-1,:2])*delta).sum(-1)/l2,0,1)
                distance=float(np.linalg.norm(xy-route[:-1,:2]-frac[:,None]*delta,axis=1).min())
                if o['object_id'] in ['obstacle','block0'] and distance>2.:
                    fixture_issues.append(dict(scene_id=r['scene_id'],object=o['object_id'],distance_to_route_m=distance,
                      status='FixtureMismatch_actor_placed_on_initial_tangent_not_curved_route',changed=False))
            for c,e in zip(r['candidates'],r['evaluations']):
                # Stale/missing were already contract-rejected, no fake geometry denominators.
                if not e['planning_feasible']:continue
                q=np.arange(0,6.0001,.025);poses=interp(c['points'],c['times_s'],q)
                collisions=[];future_collisions=[];road_bad=0
                allowed={cat['lane']}
                side='left' if c['name']=='B_left' else 'right' if c['name']=='C_right' else None
                if side in cat['sides'] and cat['sides'][side]['permitted']:allowed.add(cat['sides'][side]['lane'])
                for pose,tt in zip(poses,q):
                    x,y,yaw=pose
                    cc=np.array([x+math.cos(yaw)*offset[0]-math.sin(yaw)*offset[1],y+math.sin(yaw)*offset[0]+math.cos(yaw)*offset[1]])
                    body=Polygon(footprint(*cc,yaw,*half))
                    for o in obstacles:
                        pos=np.array(o['xy'])+tt*np.array(o['velocity'])
                        other=Polygon(footprint(*pos,o['yaw']+tt*o['yaw_rate'],o['half_length'],o['half_width']))
                        if body.intersects(other):collisions.append((o['object_id'],float(tt)))
                        ref=r['future_reference_evaluator_only'].get(o['object_id'])
                        if ref:
                            ft=np.array([v[0] for v in ref]);fp=np.array([v[1] for v in ref])
                            actual=[np.interp(tt,ft,fp[:,k]) for k in (0,1)]
                            if body.intersects(Polygon(footprint(*actual,o['yaw'],o['half_length'],o['half_width']))):
                                future_collisions.append((o['object_id'],float(tt)))
                    # Full interior body grid, different method from polygon covers.
                    for xx,yy in body_samples(*cc,yaw,*half,spacing=.4):
                        wp=cmap.get_waypoint(carla.Location(x=float(xx),y=-float(yy),z=0.),project_to_road=False,lane_type=carla.LaneType.Driving)
                        if wp is None or wp.road_id!=r['road'] or wp.lane_id not in allowed:road_bad+=1
                records.append(dict(scene_id=r['scene_id'],candidate=c['name'],accepted_by_evaluator=True,
                  dense_CV_collisions=collisions,scripted_future_collisions=future_collisions,
                  exact_map_body_grid_violations=road_bad,sample_dt=.025,body_grid_m=.4,
                  semantics='Independent sampled numerical check, not continuous mesh truth'))
    distances=[]
    for ra,a in locations.items():
        for rb,b in locations.items():
            if rb<=ra or a['split']==b['split']:continue
            aa=np.array(a['route'])[:,:2];bb=np.array(b['route'])[:,:2]
            distances.append(dict(road_a=ra,road_b=rb,split_a=a['split'],split_b=b['split'],
              minimum_sampled_route_distance_m=float(np.linalg.norm(aa[:,None]-bb[None],axis=-1).min())))
    write('reports/m7a_independent_geometry_audit.json',dict(kind='OfflineIndependentNumericalReferenceAudit',provenance=provenance(),
      accepted_candidate_checks=records,accepted_candidates=len(records),
      accepted_with_CV_collision=sum(bool(r['dense_CV_collisions']) for r in records),
      accepted_with_road_violation=sum(r['exact_map_body_grid_violations']>0 for r in records),
      accepted_with_scripted_future_collision=sum(bool(r['scripted_future_collisions']) for r in records),
      fixture_issues=fixture_issues,location_distances=distances,
      limits=['Sampled checker can detect but not prove absence of between-sample errors','OpenDRIVE same map reference not independent mesh truth','Scripted future may violate CV; forecast uncertainty not solved','No held-out protocol or positions changed after outcomes']))
    print('accepted audited',len(records),'fixture issues',len(fixture_issues))

if __name__=='__main__':main()
