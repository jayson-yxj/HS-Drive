"""Read-only CARLA lane legality survey for independent straight maneuver sites."""
import json, math
from pathlib import Path
import carla
import numpy as np
from tools.m6b_common import write, provenance


def main():
    c=carla.Client('localhost',2100);c.set_timeout(30);w=c.get_world()
    if not w.get_map().name.endswith('Town03'):w=c.load_world('Town03')
    m=w.get_map();out=[]
    for i,tf in enumerate(m.get_spawn_points()):
        wp=m.get_waypoint(tf.location);path=[wp]
        for _ in range(100):
            ns=path[-1].next(1)
            if not ns:break
            path.append(min(ns,key=lambda q:abs(q.transform.rotation.yaw-path[-1].transform.rotation.yaw)))
        if len(path)<90 or any(q.is_junction for q in path):continue
        y=np.unwrap(np.radians([q.transform.rotation.yaw for q in path]))
        if np.max(abs(y-y[0]))>.10:continue
        for side in ['left','right']:
            def legal(q):
                other=q.get_left_lane() if side=='left' else q.get_right_lane()
                marking=q.left_lane_marking if side=='left' else q.right_lane_marking
                flag=carla.LaneChange.Left if side=='left' else carla.LaneChange.Right
                return other is not None and other.lane_type==carla.LaneType.Driving and other.lane_id*q.lane_id>0 and int(q.lane_change)&int(flag) and marking.type==carla.LaneMarkingType.Broken
            if all(legal(q) for q in path[:90]):
                out.append(dict(spawn_index=i,road_id=wp.road_id,lane_id=wp.lane_id,side=side,
                    xyz=[tf.location.x,tf.location.y,tf.location.z],yaw=tf.rotation.yaw,lane_width=wp.lane_width,
                    route=[[q.transform.location.x,-q.transform.location.y,-math.radians(q.transform.rotation.yaw)] for q in path]))
    write('reports/m6b_lane_survey.json',dict(**provenance('actual_CARLA_offline_map_geometry'),source='Privileged',map=m.name,sites=out))
    print([(p['spawn_index'],p['road_id'],p['lane_id'],p['side']) for p in out])

if __name__=='__main__':main()
