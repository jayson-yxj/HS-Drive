"""Offline recorded-pose full-body audit against exact-version CARLA OpenDRIVE.

Actor OBB and Driving waypoint predicate are approximate geometry references,
not independent render/curb meshes. No CARLA tick is issued by this tool.
"""
import json,math
from pathlib import Path
import numpy as np,carla
from simulator.m6b_bypass import body_samples
from simulator.oracle_driver import footprint,polygons_intersect
from tools.m6b_common import write,provenance,asset


def main():
    ref=json.loads(Path('reports/m6b_body_reference.json').read_text())
    cmap=carla.Map('m6b_recorded_Town03',Path('runs/m6b_logs/Town03.xodr').read_text())
    bb=carla.BoundingBox(carla.Location(*ref['location']),carla.Vector3D(*ref['extent']))
    roll,pitch,yaw=ref['rotation'];bb.rotation=carla.Rotation(roll=roll,pitch=pitch,yaw=yaw)
    records=[]
    for path in sorted(Path('reports/m6b_trials').glob('final_*/result.json')):
        result=json.loads(path.read_text());rows=[json.loads(s) for s in (path.parent/'trace.jsonl').read_text().splitlines()]
        allowed=None;bypass=result['bypass']
        if bypass['status']=='FeasibleBypass':
            selected=next(c for c in bypass['candidates'] if c['side']==bypass['selected'])
            allowed=set(map(tuple,selected['allowed_lanes']))
        selected_rows=[];last=None
        for i,r in enumerate(rows):
            xy=np.array(r['measurement']['position'][:2]);angle=r['measurement']['rotation'][2]
            if last is None or np.linalg.norm(xy-last[0])>=.25 or abs(angle-last[1])>=.5 or i==len(rows)-1:
                selected_rows.append(r);last=(xy,angle)
        violations=[];collisions=[];checks=0
        for r in selected_rows:
            meas=r['measurement'];roll,pitch,yaw=meas['rotation'];tf=carla.Transform(carla.Location(*meas['position']),carla.Rotation(roll=roll,pitch=pitch,yaw=yaw))
            vertices=bb.get_world_vertices(tf);xyz=np.array([[p.x,-p.y,p.z] for p in vertices]);center=xyz.mean(0)
            heading=-math.radians(yaw+bb.rotation.yaw);c,s=math.cos(heading),math.sin(heading)
            extent=abs((xyz[:,:2]-center[:2])@np.array([[c,-s],[s,c]])).max(0)
            ego=footprint(*center[:2],heading,*extent)
            for obs in r['obstacles']:
                if polygons_intersect(ego,footprint(*obs['xy'],obs['yaw'],obs['half_length'],obs['half_width'])):
                    collisions.append(dict(frame=r['frame'],actor_id=obs['actor_id']))
            bad=[]
            for xx,yy in body_samples(*center[:2],heading,extent[0]+.2,extent[1]+.15):
                checks+=1
                wp=cmap.get_waypoint(carla.Location(x=float(xx),y=-float(yy),z=float(meas['position'][2])),project_to_road=False,lane_type=carla.LaneType.Driving)
                if wp is None or (allowed is not None and (wp.road_id,wp.lane_id) not in allowed):bad.append([float(xx),float(yy)])
            if bad:violations.append(dict(frame=r['frame'],outside_samples=len(bad),example_xy=bad[:4]))
        records.append(dict(name=path.parent.name,poses=len(selected_rows),road_queries=checks,
            footprint_outside_frames=len(violations),examples=violations[:10],OBB_intersections=collisions,
            source='Privileged',result=asset(path)))
        print(path.parent.name,len(selected_rows),len(violations),len(collisions),flush=True)
    write('reports/m6b_recorded_body_audit.json',dict(**provenance('offline_actual_CARLA_pose_OBB_and_OpenDRIVE_audit'),source='Privileged',
        map=asset('runs/m6b_logs/Town03.xodr'),body=asset('reports/m6b_body_reference.json'),trials=records,
        limits='Measured OBB offset/roll/pitch included. Poses sampled every >=.25m or .5deg and final pose; body grid .2m plus .2/.15m margin. Discrete approximation; no exact curb mesh or continuous contact proof.'))

if __name__=='__main__':main()
