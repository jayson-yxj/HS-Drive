"""Current snapshot + immutable map adapter; no Client, future or Actor API."""
import math
import numpy as np
import carla
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tools.m7a_scenarios import lane_polygon
from diagnostics.m7c_routes import continue_waypoints
from diagnostics.m7b_motion import ObservedMotion


class RoadIndex:
    def __init__(self, cmap):
        self.map = cmap
        groups = {}
        for w in cmap.generate_waypoints(.5):
            if w.lane_type == carla.LaneType.Driving:
                groups.setdefault((w.road_id, w.lane_id, w.section_id), []).append(w)
        self.polygons = {k: lane_polygon(sorted(ws, key=lambda w: w.s)) for k, ws in groups.items()}

    def waypoint(self, xy, z, project=False):
        return self.map.get_waypoint(carla.Location(x=float(xy[0]), y=-float(xy[1]), z=float(z)),
                                    project_to_road=project, lane_type=carla.LaneType.Driving)

    @staticmethod
    def key(w):
        return (w.road_id, w.lane_id, w.section_id)

    def region(self, keys):
        return unary_union([self.polygons[k] for k in sorted(keys) if k in self.polygons])

    def ego_support(self, inp):
        ws = [self.waypoint(p[:2], inp.ego_z) for p in inp.ego_route]
        keys = {self.key(w) for w in ws if w is not None}
        sides = {}
        for side, flag, back in [('left', carla.LaneChange.Left, carla.LaneChange.Right),
                                 ('right', carla.LaneChange.Right, carla.LaneChange.Left)]:
            ns = [(w.get_left_lane() if side == 'left' else w.get_right_lane()) if w else None for w in ws]
            if not ns or not all(n and w and n.lane_type == carla.LaneType.Driving and n.lane_id*w.lane_id > 0 for n,w in zip(ns,ws)):
                continue
            permitted = all(int(w.lane_change)&int(flag) and int(n.lane_change)&int(back)
                            and (w.left_lane_marking if side=='left' else w.right_lane_marking).type==carla.LaneMarkingType.Broken
                            for w,n in zip(ws,ns))
            sides[side] = dict(offset=(1 if side=='left' else -1)*.5*(ws[0].lane_width+ns[0].lane_width),
                               permitted=bool(permitted), keys={self.key(n) for n in ns})
        return keys, sides, sum(w is None for w in ws)

    def observations(self, inp):
        obs, routes, unknown, regions = [], [], [], {}
        for b, meta in zip(inp.actors, inp.actor_metadata):
            if meta[1] != 'Vehicle':
                unknown.append(dict(object_id=b.object_id, reason='ActorCategoryWithoutDeclaredMotionCorridor', category=meta[1]))
                continue
            w = self.waypoint(b.xy, inp.ego_z)
            if w is None:
                unknown.append(dict(object_id=b.object_id, reason='CurrentActorCenterNotDriving'));continue
            behind = w.previous(5.)
            start = behind[0] if len(behind)==1 else w
            try:
                conn = continue_waypoints(start, required_length=80., max_extra=80.)
                s0, distance = conn.project(np.array(b.xy))
                obs.append(ObservedMotion(b.object_id, conn.route, s0, 'road_vehicle'))
                keys = {(n['road'],n['lane'],n['section']) for n in conn.nodes}
                regions[b.object_id] = self.region(keys)
                routes.append(dict(object_id=b.object_id, reason=conn.stop_reason, status=conn.status,
                                   projection_error_m=distance, length_m=conn.length, s0=s0,
                                   transitions=conn.transitions, keys=sorted(keys), intent=None))
            except ValueError as e:
                unknown.append(dict(object_id=b.object_id, reason=str(e)))
        return obs, routes, unknown, regions
