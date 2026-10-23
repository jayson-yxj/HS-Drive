"""One opt-in optimization: snapshot-scoped polygon reuse, identical queries."""
import numpy as np
from shapely.geometry import Polygon
from diagnostics.m7a_planning import center
from diagnostics.m7b_motion import interpolate
from simulator.oracle_driver import footprint


class SnapshotPolygons:
    """Lifetime is ONE immutable snapshot; never stored in cross-frame cache."""
    def __init__(self, scene, hypotheses):
        self.scene = scene
        boxes = {b.object_id: b for b in scene.boxes}
        self.actors = {}
        for h in hypotheses:
            b = boxes[h.object_id]
            self.actors[(h.object_id, h.name)] = tuple(
                Polygon(footprint(*p[:2], p[2], b.half_length, b.half_width))
                for p in h.poses)
        self.ego = {}

    def ego_polygons(self, candidate, times):
        key = (candidate.name, candidate.points, tuple(times))
        if key not in self.ego:
            ego = interpolate(candidate.points, candidate.times_s, times)
            polygons = tuple(Polygon(footprint(*center(p, self.scene.body_center_offset),
                             p[2], *self.scene.body_half_extent)) for p in ego)
            self.ego[key] = (ego, polygons)
        return self.ego[key]


def stopping_reserve_reused(candidate, scene, hypotheses, route, polygons):
    """Same early-exit order/minimum as M7C; only polygons are reused."""
    if polygons.scene is not scene:
        raise ValueError('snapshot_identity_mismatch')
    failures = []
    minimum = None
    for h in hypotheses:
        ego, bodies = polygons.ego_polygons(candidate, h.times)
        others = polygons.actors[(h.object_id, h.name)]
        for ep, body, ob, t in zip(ego, bodies, others, h.times):
            margin = body.distance(ob) - (3. + ep[3]**2/(2*1.8) + ep[3]*1.8/2.5)
            minimum = margin if minimum is None else min(minimum, margin)
            if margin < -1e-8:
                failures.append(dict(actor=h.object_id, hypothesis=h.name,
                                     time_s=float(t), reserve_deficit_m=float(-margin)))
                break
    endpoint = np.array(candidate.points[-1])
    s, _ = route.project(endpoint[:2])
    remaining = route.length-s-scene.body_half_extent[0]-scene.body_center_offset[0]
    required = 3+endpoint[3]**2/(2*1.8)+endpoint[3]*1.8/2.5
    if remaining < required:
        failures.append(dict(reason='route_tail_stop_reserve', remaining=remaining, required=required))
    return dict(passed=not failures, failures=failures, minimum_actor_reserve_m=minimum,
                route_tail_m=remaining, stop_distance_required_m=required,
                meaning='Conservative finite-hypothesis planning reserve; FeasibleToExecute=NotValidated')
