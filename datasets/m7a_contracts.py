"""Opt-in immutable offline contracts. No conversion to an actuator API."""
from dataclasses import dataclass, fields
from typing import ClassVar
import math

WORLD = 'hs_world_x_y_left_z_up_m'
EGO = 'hs_ego_x_forward_y_left_z_up_m'


@dataclass(frozen=True)
class Stamp:
    frame_id: int
    timestamp_s: float
    valid_until_s: float
    coordinate: str
    version: str = 'm7a.v1'

    def validate(self, now):
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('invalid_frame')
        if self.version != 'm7a.v1' or self.coordinate not in (WORLD, EGO):
            raise ValueError('coordinate_or_version')
        if not all(math.isfinite(v) for v in (self.timestamp_s, self.valid_until_s, now)):
            raise ValueError('nonfinite_time')
        if not self.timestamp_s <= now <= self.valid_until_s:
            raise ValueError('stale_or_future')


@dataclass(frozen=True)
class Box:
    object_id: str
    xy: tuple
    yaw: float
    half_length: float
    half_width: float
    velocity: tuple = (0., 0.)
    yaw_rate: float = 0.

    def validate(self):
        if len(self.xy) != 2 or len(self.velocity) != 2 or not all(math.isfinite(x) for x in (*self.xy, self.yaw, self.half_length, self.half_width, *self.velocity, self.yaw_rate)):
            raise ValueError('invalid_box')
        if self.half_length <= 0 or self.half_width <= 0:
            raise ValueError('invalid_box_extent')


@dataclass(frozen=True)
class OracleSceneState:
    stamp: Stamp
    ego_xy_yaw_speed: tuple
    boxes: tuple
    body_half_extent: tuple = (2.4, 1.0)
    body_center_offset: tuple = (0., 0.)
    source: ClassVar[str] = 'Privileged'
    reference: ClassVar[str] = 'planar_OBB_not_render_mesh'

    def validate(self, now):
        self.stamp.validate(now)
        if self.stamp.coordinate != WORLD or len(self.ego_xy_yaw_speed) != 4 or not all(math.isfinite(v) for v in self.ego_xy_yaw_speed):
            raise ValueError('oracle_state_coordinate')
        if not isinstance(self.boxes, tuple) or any(type(x) is not Box for x in self.boxes):
            raise TypeError('immutable_value_boxes_required')
        for box in self.boxes:
            box.validate()
        if len(self.body_half_extent) != 2 or min(self.body_half_extent) <= 0 or not all(math.isfinite(v) for v in (*self.body_half_extent, *self.body_center_offset)):
            raise ValueError('invalid_ego_body')


@dataclass(frozen=True)
class PlanningRequest:
    scene: OracleSceneState
    route: tuple
    scene_id: str
    goal_xy_yaw: tuple
    source: ClassVar[str] = 'Privileged_OfflinePlanning'

    def validate(self, now):
        if type(self.scene) is not OracleSceneState:
            raise TypeError('oracle_type_required_no_visual_cast')
        self.scene.validate(now)
        if not isinstance(self.route, tuple) or len(self.route) < 2 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in self.route):
            raise ValueError('invalid_route')


@dataclass(frozen=True)
class CandidateTrajectory:
    stamp: Stamp
    name: str
    points: tuple  # world x,y,yaw,speed, includes t=0
    times_s: tuple  # relative to ORIGINAL snapshot
    guide_path: tuple
    lane_change_permitted: bool = True
    source: ClassVar[str] = 'Privileged_OfflineCandidate_NotControl'


@dataclass(frozen=True)
class PlanningEvaluation:
    candidate: str
    planning_feasible: bool
    reasons: tuple
    metrics: tuple
    vehicle_executed_successfully: str = 'NotRun'
    source: ClassVar[str] = 'OfflineEvaluator'


@dataclass(frozen=True)
class RGBSpatialEvidence:
    stamp: Stamp
    sample_id: str
    hypotheses: tuple
    model_sha256: str
    qualification: str = 'CandidateNoGo'
    source: ClassVar[str] = 'RGB_estimate_OfflineShadow'
    safety_free: ClassVar[bool] = False
    spatial_evidence_v2: ClassVar[str] = 'surface_and_free_ray_nonexclusive_not_safe_free'


@dataclass(frozen=True)
class DetectedObjectHypothesis:
    camera: str
    candidate_id: str
    region: tuple
    position_ego_xyz_m: tuple | None
    source: ClassVar[str] = 'RGB_estimate_OfflineShadow'
    reference: ClassVar[str] = 'ActorBoxCenter_estimate_not_contact'
    qualification: ClassVar[str] = 'CandidateNoGo'
    safety_free: ClassVar[bool] = False


@dataclass(frozen=True)
class Unknown:
    reason: str
    source: ClassVar[str] = 'Unknown'
    safety_free: ClassVar[bool] = False


def reject_control_export(value):
    """There is deliberately no M7-A normal-control adapter."""
    raise TypeError('M7A_offline_only_original_control_API_unchanged')
