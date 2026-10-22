"""Non-controlling research envelopes. Planner consumes explicit provenance domains."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpatialEnvelope:
    schema: str
    source: str
    frame_id: int
    timestamp_s: float
    valid_until_s: float
    coordinate: str = "hs_ego_x_forward_y_left_z_up_m"
    control_authorized: bool = False

    def validate(self, now_s, *, consumer="shadow"):
        if self.source not in {
            "oracle_environment",
            "rgb_bev_evidence",
            "rgb_object_candidates",
        }:
            raise ValueError("unknown provenance")
        if not all(
            math.isfinite(x) for x in [self.timestamp_s, self.valid_until_s, now_s]
        ):
            raise ValueError("nonfinite time")
        if (
            self.frame_id < 0
            or self.valid_until_s < self.timestamp_s
            or now_s < self.timestamp_s
            or now_s > self.valid_until_s
        ):
            raise ValueError("stale/future frame")
        if self.coordinate != "hs_ego_x_forward_y_left_z_up_m":
            raise ValueError("coordinate")
        if self.control_authorized or consumer != "shadow":
            raise ValueError(
                "M5-O envelopes are shadow-only; Oracle controller uses its existing separate API"
            )


@dataclass(frozen=True)
class ObjectEstimate:
    candidate_id: str
    status: str
    position_ego_xyz_m: tuple | None
    proxy_error_m: float | None
    reference: str = "simulator_actor_box_center_estimate_not_contact"
    safety_free: bool = False

    def validate(self):
        if self.status not in {"DetectedNoLocation", "DetectedWithEstimate"}:
            raise ValueError("status")
        if self.safety_free:
            raise ValueError("unknown/object evidence is never free space")
        if self.status == "DetectedNoLocation" and self.position_ego_xyz_m is not None:
            raise ValueError("rejected output cannot expose accepted position")
        if self.status == "DetectedWithEstimate" and (
            self.position_ego_xyz_m is None
            or len(self.position_ego_xyz_m) != 3
            or not all(math.isfinite(x) for x in self.position_ego_xyz_m)
        ):
            raise ValueError("position")
        if self.proxy_error_m is not None and (
            not math.isfinite(self.proxy_error_m) or self.proxy_error_m < 0
        ):
            raise ValueError("risk")
