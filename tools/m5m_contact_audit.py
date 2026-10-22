"""Audit native visible bottoms, rendered road planes and RGB-only hypotheses."""

import argparse, json, math
from pathlib import Path
import numpy as np
from tools.m5l_common import rows, cache_index, write, asset, summary
from tools.m5f_common import pixel_label, input_path
from diagnostics.m5l_objects import components, locate, match, plane_intersection
from diagnostics.m5m_contact import fit_plane, intersect, reference_status
from models.m5m_selective import predict
from tools.m5l_object_eval import errors
from geometry.reference import unproject_pixels_to_optical, transform_points
from simulator.recorder import file_hash


def points(u, v, depth, K, T):
    return transform_points(unproject_pixels_to_optical(u, v, depth, K), T)


def audit_rows(rs, ix, historical=None):
    records = []
    detections = []
    references = []
    outputs = []
    hist = (
        {}
        if historical is None
        else {
            (r["sample_id"], r["seed"], r["camera"], r["actor_id"]): r
            for r in historical
        }
    )
    for row in rs:
        raw = Path(row["raw_frame"]).parent.parent
        frame = json.loads(Path(row["raw_frame"]).read_text())
        inp = json.loads(input_path(row).read_text())
        d = pixel_label(row)
        z = np.load(ix[row["sample_id"]])
        pose = np.array(frame["T_world_from_ego"])
        pitch = math.degrees(math.asin(np.clip(pose[2, 0], -1, 1)))
        for ci, cam in enumerate(row["camera_names"]):
            K = np.array(inp["intrinsics"][ci])
            T = np.array(inp["cam_to_ego"][ci])
            sem = np.load(raw / frame["cameras"][cam]["instance"])
            semantic = sem["semantic"]
            instance = sem["instance"]
            depth = d["depth"][ci]
            truth = []
            meta = []
            refs = []
            for a in row["actors"][ci]["actors"]:
                ids = [
                    r["render_id"] for r in a["identity_candidates"] if r["accepted"]
                ]
                native = (
                    np.isin(instance, ids)
                    & np.isfinite(depth)
                    & (depth >= 1.5)
                    & (depth < 80)
                )
                m = d["actor"][ci] == a["local_id"]
                if native.sum() < 3:
                    continue
                vv, uu = np.where(native)
                bottom = int(vv.max())
                u = float(np.median(uu[vv >= bottom - 1]))
                v = float(bottom)
                uv = np.array([u, v])
                contact = points(
                    np.array([u]),
                    np.array([v]),
                    np.array([depth[int(v), int(round(u))]]),
                    K,
                    T,
                )[0]
                # Median coordinate may fall between disconnected surfaces: use actual bottom pixels.
                vm, um = np.where(native & (np.indices(native.shape)[0] >= bottom - 1))
                bp = points(um, vm, depth[vm, um], K, T)
                contact = np.median(bp, axis=0)
                y0 = max(0, int(vv.min()) - 8)
                y1 = min(384, bottom + 50)
                x0 = max(0, int(uu.min()) - 50)
                x1 = min(640, int(uu.max()) + 51)
                road = np.zeros_like(native)
                road[y0:y1, x0:x1] = True
                road &= (
                    np.isin(semantic, [1, 24])
                    & np.isfinite(depth)
                    & (depth < 80)
                    & (depth > 1.5)
                )
                rv, ru = np.where(road)
                rp = points(ru, rv, depth[rv, ru], K, T)
                rp = rp[
                    np.linalg.norm(rp[:, :2] - np.array(a["center"][:2]), axis=1) < 5
                ]
                plane = fit_plane(rp)
                aa = next(t for t in frame["actors"] if t["actor_id"] == a["actor_id"])
                box = np.array(aa["corners_hs_ego_m"])
                center = np.array(a["center"])
                road_z = (
                    None
                    if plane is None
                    else float(np.dot(plane["coefficients"], [*center[:2], 1]))
                )
                gap = None if road_z is None else float(box[:, 2].min() - road_z)
                truncated = bool(
                    uu.min() == 0 or uu.max() == 639 or vv.min() == 0 or bottom == 383
                )
                # Occlusion proxy: projected AABB lower edge vs native silhouette; never exact mesh visibility.
                opt = (np.linalg.inv(T) @ np.c_[box, np.ones(8)].T).T[:, :3]
                project = (K @ opt.T).T
                project = project[:, :2] / project[:, 2:]
                projected_bottom = (
                    float(project[opt[:, 2] > 0, 1].max())
                    if (opt[:, 2] > 0).any()
                    else None
                )
                ref = dict(
                    sample_id=row["sample_id"],
                    group=row["group"],
                    split=row["split"],
                    camera=cam,
                    actor_id=a["actor_id"],
                    type=(
                        "vehicle"
                        if a["type_id"].startswith("vehicle.")
                        else "person" if a["type_id"].startswith("walker.") else "cone"
                    ),
                    type_id=a["type_id"],
                    distance_m=a["distance_m"],
                    configuration=row["configuration"],
                    ego_pitch_deg=pitch,
                    native_visible_pixels=int(native.sum()),
                    strict_visible_pixels=int(m.sum()),
                    native_bottom_uv=uv.tolist(),
                    native_bottom_xyz=contact.tolist(),
                    bottom_z_spread_m=float(np.ptp(bp[:, 2])),
                    native_truncated=truncated,
                    projected_aabb_bottom_v=projected_bottom,
                    bbox_bottom_visibility_gap_px=(
                        None if projected_bottom is None else projected_bottom - bottom
                    ),
                    rendered_local_road=plane,
                    road_z_at_actor_center=road_z,
                    fixed_plane_height_error_m=(
                        None if road_z is None else abs(road_z + 0.25)
                    ),
                    aabb_bottom_road_gap_m=gap,
                    placement_flag=(
                        "unresolved"
                        if gap is None
                        else (
                            "possible_floating"
                            if gap > 0.25
                            else (
                                "possible_penetration"
                                if gap < -0.25
                                else "box_near_road"
                            )
                        )
                    ),
                    status=reference_status(
                        plane=plane, visible_pixels=int(native.sum())
                    ),
                    gold_anatomy_verified=False,
                    collision_agreement_verified=False,
                    reference="Native rendered silhouette and local rendered Road plane; AABB is proxy, no wheel/foot identity. No physical contact Gold.",
                    center=center.tolist(),
                )
                references.append(ref)
                truth.append(native)
                meta.append(a)
                refs.append(ref)
            for seed in [101, 102]:
                q = z[f"pixel_{seed}"].astype(np.float32)
                cand = components(q[ci])
                pred = predict(q[ci], K, T, timestamp=frame["timestamp_s"], camera=cam)
                pairs, fp, fn = match(cand, truth)
                pairmap = {j: (i, iou) for i, j, iou in pairs}
                outputs.extend(
                    dict(sample_id=row["sample_id"], seed=seed, camera=cam, **o)
                    for o in pred
                )
                detections.append(
                    dict(
                        sample_id=row["sample_id"],
                        group=row["group"],
                        seed=seed,
                        camera=cam,
                        visible=len(truth),
                        candidates=len(cand),
                        matched=len(pairs),
                        undetected=len(fn),
                        unmatched_candidates=len(fp),
                        unmatched_interpretation="Not all physical false positives: unlabelled objects remain Unknown",
                    )
                )
                for j, (a, ref) in enumerate(zip(meta, refs)):
                    r = dict(
                        ref,
                        seed=seed,
                        detected=j in pairmap,
                        baseline_valid=False,
                        rule_eligible=False,
                        confidence=0.0,
                    )
                    if j in pairmap:
                        idx, iou = pairmap[j]
                        cc = cand[idx]
                        o = pred[idx]
                        p, why = locate(cc, K, T)
                        uv = cc["uv"]
                        center = np.array(a["center"])
                        bottom = np.array(ref["native_bottom_xyz"])
                        r.update(
                            iou=iou,
                            output=o,
                            baseline_valid=p is not None,
                            baseline_reason=why,
                            rule_eligible=o["conditional_status"]
                            == "ConditionalPosition",
                            confidence=o["confidence"],
                            pred_bottom_minus_visible_px=(
                                uv - np.array(ref["native_bottom_uv"])
                            ).tolist(),
                            merged_actor_count=sum(
                                (cc["mask"] & t).sum() >= 3 for t in truth
                            ),
                            native_coverage=float(
                                (cc["mask"] & truth[j]).sum() / truth[j].sum()
                            ),
                        )
                        if p is not None:
                            r.update(
                                plane_center=errors(p, center),
                                plane_bottom=errors(p, bottom),
                            )
                        oracle = (
                            intersect(uv, K, T, ref["rendered_local_road"])
                            if ref["rendered_local_road"]
                            else None
                        )
                        gtbottom = plane_intersection(
                            np.array(ref["native_bottom_uv"]), K, T
                        )[0]
                        exact = (
                            intersect(
                                np.array(ref["native_bottom_uv"]),
                                K,
                                T,
                                ref["rendered_local_road"],
                            )
                            if ref["rendered_local_road"]
                            else None
                        )
                        for name, pt in [
                            ("oracle_plane_predbottom", oracle),
                            ("fixed_plane_gtbottom", gtbottom),
                            ("oracle_plane_gtbottom", exact),
                        ]:
                            if pt is not None:
                                r[name] = errors(pt, bottom)
                        old = hist.get((row["sample_id"], seed, cam, a["actor_id"]))
                        if (
                            old is not None
                            and "uv" in old
                            and np.allclose(uv, old["uv"])
                        ):
                            for name in ["dense_expected_center", "dense_mode_center"]:
                                r[name] = old[name]
                    records.append(r)
        print(row["sample_id"], flush=True)
    return dict(
        actors=records, detections=detections, references=references, outputs=outputs
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    old = json.loads(Path(f"runs/m5l_objects_{a.split}.json").read_text())["actors"]
    result = audit_rows(rows(a.split), cache_index(a.split), old)
    write(f"runs/m5m_contact_{a.split}.json", result)
    write(
        f"reports/m5m_contact_{a.split}.json",
        dict(
            kind="offline_actual_recorded_CARLA_renderer_geometry_and_cached_RGB",
            raw=asset(f"runs/m5m_contact_{a.split}.json"),
            source_sha256=file_hash(__file__),
            native_actor_views=len(result["references"]),
            gold=sum(r["status"] == "Gold" for r in result["references"]),
            candidate=sum(r["status"] == "Candidate" for r in result["references"]),
            unknown=sum(r["status"] == "Unknown" for r in result["references"]),
            limits="Native associated surface denominator includes thin actors erased by strict mask. Matching differs from M5L; frozen M5L original metrics retained separately. Native bottom, AABB and rendered road are not exact contact.",
        ),
    )


if __name__ == "__main__":
    main()
