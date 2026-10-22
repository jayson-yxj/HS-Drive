"""Source-explicit data contracts, without a connection to vehicle actuation.

These types are validation boundaries, not an authentication mechanism. Runtime
still uses its existing privileged API; no visual consumer is granted control.
"""
from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass(frozen=True)
class Envelope:
    frame_id: int
    timestamp_s: float
    valid_until_s: float
    coordinate: str = 'hs_ego_x_forward_y_left_z_up_m'
    version: str = 'm6a.v1'
    source: ClassVar[str] = 'unspecified'

    def validate(self, now_s, *, consumer='offline_shadow'):
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('invalid frame')
        if not all(math.isfinite(v) for v in (self.timestamp_s, self.valid_until_s, now_s)):
            raise ValueError('nonfinite timestamp')
        if not self.timestamp_s <= now_s <= self.valid_until_s:
            raise ValueError('future or stale original timestamp')
        if self.version != 'm6a.v1' or self.coordinate != 'hs_ego_x_forward_y_left_z_up_m':
            raise ValueError('unsupported version or coordinate')
        if consumer not in ('offline_shadow', 'oracle_planner'):
            raise ValueError('unknown consumer')
        if consumer == 'oracle_planner' and type(self) is not OracleSceneState:
            raise ValueError('visual/unknown evidence cannot feed Oracle control')

    def to_record(self):
        from dataclasses import asdict
        return dict(schema=type(self).__name__, source=self.source, **asdict(self))


@dataclass(frozen=True)
class OracleSceneState(Envelope):
    source: ClassVar[str] = 'Privileged'
    reference: str = 'CARLA_snapshot_actor_box_and_map_not_render_mesh'


@dataclass(frozen=True)
class RGBSpatialEvidence(Envelope):
    source: ClassVar[str] = 'RGB_estimate_OfflineShadow'
    evidence_contract: str = 'SpatialEvidenceV2_surface_and_free_ray_nonexclusive'
    space_shape: tuple = (4, 160, 120)


@dataclass(frozen=True)
class DetectedObjectHypothesis(Envelope):
    source: ClassVar[str] = 'RGB_estimate_OfflineShadow'
    position_xyz_m: tuple | None = None
    uncertainty_radius_m: float | None = None
    reference: str = 'actor_box_center_estimate_not_contact'

    def validate(self, now_s, *, consumer='offline_shadow'):
        super().validate(now_s, consumer=consumer)
        if self.position_xyz_m is not None and (len(self.position_xyz_m) != 3 or
                not all(math.isfinite(v) for v in self.position_xyz_m)):
            raise ValueError('invalid position')
        if self.uncertainty_radius_m is not None and (not math.isfinite(self.uncertainty_radius_m)
                                                     or self.uncertainty_radius_m < 0):
            raise ValueError('invalid uncertainty')


@dataclass(frozen=True)
class Unknown(Envelope):
    source: ClassVar[str] = 'Unknown_not_safe_free'
    reason: str = 'insufficient_evidence'

    @property
    def safety_free(self):
        return False
