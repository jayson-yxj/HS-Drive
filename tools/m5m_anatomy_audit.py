"""Audit recorded wheel/foot coordinate support and renderer visibility, without inferring Gold."""

import argparse, json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary
from tools.m5f_common import input_path, pixel_label
from geometry.reference import transform_points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    a = p.parse_args()
    rs = json.loads(
        Path(
            f"/root/gpufree-data/hs-drive-data/processed/m5m_{a.phase}_pixels/index.json"
        ).read_text()
    )["records"]
    out = []
    for row in rs:
        f = json.loads(Path(row["raw_frame"]).read_text())
        raw = Path(row["raw_frame"]).parent.parent
        pose = np.array(f["T_world_from_ego"])
        inv = np.linalg.inv(pose)
        inp = json.loads(input_path(row).read_text())
        d = pixel_label(row)
        for actor in f["actors"]:
            if (
                actor["is_ego"]
                or np.linalg.norm(np.array(actor["corners_hs_ego_m"]).mean(0)[:2]) > 80
            ):
                continue
            world = np.array(actor["world_matrix"])
            points = []
            coordinate_check = {}
            wheel = actor.get("wheels") or []
            if wheel:
                wp = np.array([w["position_cm"] for w in wheel]) * 0.01
                # SDK wheel positions are world centimetres; explicitly compare alternative conventions.
                coordinate_check = dict(
                    world_centroid_distance_m=float(
                        np.linalg.norm(wp.mean(0) - world[:3, 3])
                    ),
                    local_if_misread_world_centroid_distance_m=float(
                        np.linalg.norm(
                            transform_points(wp, world).mean(0) - world[:3, 3]
                        )
                    ),
                    interpretation="World centimeters candidate; radius is physics wheel, not visual mesh/tread/contact patch",
                )
                for w, pw in zip(wheel, wp):
                    points.append(
                        (
                            "physics_wheel_bottom",
                            pw - np.array([0, 0, w["radius_cm"] * 0.01]),
                        )
                    )
            for b in actor.get("foot_bones") or []:
                points.append((b["name"], np.array(b["world"])[:3, 3]))
            refs = actor.get("contact_reference") or {}
            roads = [
                h["xyz"]
                for probe in refs.get("probes", [])
                for h in probe
                if h["label"] in [1, 24]
            ]
            views = []
            for ci, cam in enumerate(row["camera_names"]):
                K = np.array(inp["intrinsics"][ci])
                T = np.array(inp["cam_to_ego"][ci])
                meta = next(
                    (
                        x
                        for x in row["actors"][ci]["actors"]
                        if x["actor_id"] == actor["actor_id"]
                    ),
                    None,
                )
                if meta is None:
                    continue
                inst = np.load(raw / f["cameras"][cam]["instance"])["instance"]
                ids = [
                    r["render_id"] for r in meta["identity_candidates"] if r["accepted"]
                ]
                for name, pw in points:
                    hs = (inv @ np.r_[pw * [1, -1, 1], 1])[:3]
                    opt = (np.linalg.inv(T) @ np.r_[hs, 1])[:3]
                    uv = K @ opt
                    uv = uv[:2] / uv[2] if opt[2] > 0 else np.array([-1, -1])
                    u, v = np.rint(uv).astype(int)
                    inside = 0 <= u < 640 and 0 <= v < 384
                    depth = None
                    identity = False
                    if inside:
                        dep = d["depth"][
                            ci, max(0, v - 2) : v + 3, max(0, u - 2) : u + 3
                        ]
                        im = inst[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
                        m = np.isin(im, ids)
                        identity = bool(m.any())
                        depth = (
                            float(np.min(abs(dep[m] - opt[2]))) if identity else None
                        )
                    roadgap = min([abs(pw[2] - q[2]) for q in roads] or [float("inf")])
                    views.append(
                        dict(
                            camera=cam,
                            name=name,
                            world=pw.tolist(),
                            uv=uv.tolist(),
                            inside=inside,
                            associated_instance_near=identity,
                            render_depth_residual_m=depth,
                            nearest_probe_height_gap_m=(
                                roadgap if np.isfinite(roadgap) else None
                            ),
                            contact_verified=False,
                        )
                    )
            out.append(
                dict(
                    sample_id=row["sample_id"],
                    group=row["group"],
                    split=row["split"],
                    type_id=actor["type_id"],
                    coordinate_check=coordinate_check,
                    anatomical_points=len(points),
                    views=views,
                    status="Candidate" if points else "Unknown",
                    gold=False,
                    reason="Physics wheel radius/foot bone origin not calibrated visual sole/tread; neighboring renderer agreement cannot certify exact visible contact patch. No measured stable-pose time series.",
                )
            )
    write(f"runs/m5m_anatomy_{a.phase}.json", out)
    write(
        f"reports/m5m_anatomy_{a.phase}.json",
        dict(
            kind="actual_CARLA_anatomy_metadata_plus_offline_projection",
            actors=len(out),
            gold=0,
            with_anatomy=sum(r["anatomical_points"] > 0 for r in out),
            visible_anatomy_proxy=sum(
                v["inside"]
                and v["associated_instance_near"]
                and v["render_depth_residual_m"] is not None
                and v["render_depth_residual_m"] <= 0.1
                for r in out
                for v in r["views"]
            ),
            positions_by_split={
                s: len({r["group"] for r in out if r["split"] == s})
                for s in ["train", "val", "test"]
            },
            raw=asset(f"runs/m5m_anatomy_{a.phase}.json"),
            limits="No Gold promotion from bone or physics wheel alone. This feasibility audit cannot supply enough independently validated positives/negatives for Contact Head.",
        ),
    )


if __name__ == "__main__":
    main()
