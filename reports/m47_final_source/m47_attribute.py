"""Join immutable ray and collision audits without claiming exact mesh truth."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash


def main():
    target=Path("reports/m47_false_free_attribution_verified.json")
    if target.exists():raise FileExistsError(target)
    x=json.loads(Path("reports/m47_false_free_audit.json").read_text())
    collision=json.loads(Path("reports/m47_collision_audit.json").read_text())
    by={r["id"]:r for r in collision["records"]}
    for r in x["records"]:
        other=by[r["id"]];inside=[]
        for ray in other["free_rays"]:
            for h in ray["early_hits"]:
                point=np.asarray(h["point_hs"]).reshape(3)
                if np.all(point>=r["bounds_xyz_m"][0]) and np.all(point<r["bounds_xyz_m"][1]):inside.append(h)
        r["collision_evidence"]=dict(source_id=other["id"],early_hits_inside_cell=inside,
            early_ray_hit_elsewhere=other["dangerous_ray_contradiction_candidate"] and not bool(inside),
            render_collision_agreement_count=other["render_collision_agreement_count"])
        r["classification"]="coarse_cell_partial_occupancy_with_ray_evidence"
        r["dangerous_false_free"]="unresolved for driving; no collision query contradiction inside audited cell" if not inside else "collision geometry contradiction inside cell"
        r["same_physical_surface"]="Not proven. Sparse carving segment and heldout endpoints are spatially separated; identical instance only identifies an object, not a face."
        r["confidence"]="High for surface/ray coexistence. Insufficient to certify whole-cell or driving safety."
    x["collision_audit_sha256"]=file_hash("reports/m47_collision_audit.json")
    x["observations"]=dict(minimum_surface_to_free_segment_m=min(r["minimum_point_to_free_segment_m"] for r in x["records"]),
        near_height_boundary_lt_0_1m=sum(r["min_height_boundary_distance_m"]<.1 for r in x["records"]),
        multi_camera_cells=sum(r["camera_overlap"] for r in x["records"]),
        collision_hits_inside_cells=sum(bool(r["collision_evidence"]["early_hits_inside_cell"]) for r in x["records"]),
        whole_cell_free_disproven=len(x["records"]),dangerous_driving_errors_proven=None)
    atomic_json(target,x)


if __name__=="__main__":main()
