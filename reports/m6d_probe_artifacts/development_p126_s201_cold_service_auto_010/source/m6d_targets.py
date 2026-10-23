"""Distinct navigation goal and physical obstacle constraint; no surrogate goals."""
from dataclasses import dataclass,asdict
import math

@dataclass(frozen=True)
class StopTargetContext:
    goal_signed_remaining_m: float
    goal_route_remaining_m: float
    obstacle_safe_stop_remaining_m: float | None
    active_stop_target: str
    active_stop_reason: str
    required_clearance_m: float
    estimated_stop_capability: dict
    timestamp_s: float
    frame_id: int
    target_source: str='Privileged'
    coordinate_frame: str='hs_world_x_y_left_z_up_m'

    @property
    def active_remaining_m(self):
        if self.active_stop_target=='navigation_goal':return self.goal_route_remaining_m
        if self.active_stop_target=='obstacle_constraint':return self.obstacle_safe_stop_remaining_m
        return None


def separate_targets(geometry,frame_id,capability=None,fault=None):
    goal=float(geometry['goal_signed_distance_m']);arc=float(geometry['route_remaining_s'])
    gap=geometry['obstacle_obb_clearance_m'];obstacle=None if gap is None else float(gap)-3.
    if not all(math.isfinite(x) for x in [goal,arc,geometry['timestamp']]+([] if obstacle is None else [obstacle])):raise ValueError('InvalidStopTarget')
    if geometry['source']!='Privileged':raise ValueError('WrongTargetSource')
    if fault:kind,why='forced_degradation',str(fault)
    elif obstacle is not None and obstacle<=arc:kind,why='obstacle_constraint','physical_OBB_reserve_3m'
    else:kind,why='navigation_goal','original_StopGoal'
    return StopTargetContext(goal,arc,obstacle,kind,why,3.,capability or {},float(geometry['timestamp']),int(frame_id))


class TargetTracker:
    def __init__(self):self.previous=None;self.transitions=[]
    def update(self,target):
        key=(target.active_stop_target,target.active_stop_reason)
        if key!=self.previous:
            self.transitions.append(dict(previous=self.previous,current=asdict(target),reason=target.active_stop_reason))
            self.previous=key
