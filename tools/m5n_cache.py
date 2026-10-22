"""Freeze RGB-only candidate features; supervision written separately after inference."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch
from tools.m5n_common import rows, write, asset
from tools.m5f_common import inputs, input_path, pixel_label
from tools.m5i_data import load
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5l_p4 import sample_map
from models.m5n_direct import feature_vector
from diagnostics.m5l_objects import components, match, locate
from datasets.m5n_reference import actor_reference, synchronized
from training.runtime import prepare_cuda
from simulator.recorder import file_hash
from geometry.reference import unproject_pixels_to_optical, transform_points


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    dest = Path(f"runs/m5n_cache_{a.split}")
    dest.mkdir(exist_ok=False)
    prepare_cuda()
    torch.set_num_threads(2)
    cfg = json.loads(Path("configs/m5j_models.json").read_text())
    mobile = {r["seed"]: load(r)[0] for r in cfg["pixel"] if r["name"] == "mobile_gt"}
    net = encoder_load()
    candidates = []
    actors = []
    labels = []
    X = []
    reference_audit = []
    frame_audit = []
    timings = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for row in rows(a.split):
        t = time.perf_counter()
        x = inputs(row)
        frame = json.loads(Path(row["raw_frame"]).read_text())
        inp = json.loads(input_path(row).read_text())
        assert synchronized(frame, inp), row["sample_id"]
        raw = Path(row["raw_frame"]).parent.parent
        pose = np.array(frame["T_world_from_ego"])
        d = pixel_label(row)
        pitch = float(np.degrees(np.arcsin(np.clip(pose[2, 0], -1, 1))))
        dense = cache_p8(net, x)
        rs = {}
        truth = {}
        refs = {}
        metadata = {}
        unknown_association = 0
        # Explicit reference sidecar, never fed to mobile, features or head.
        for ci, cam in enumerate(row["camera_names"]):
            K = np.array(inp["intrinsics"][ci])
            T = np.array(inp["cam_to_ego"][ci])
            inst = np.load(raw / frame["cameras"][cam]["instance"])
            idsimg = inst["instance"]
            truth[ci] = []
            refs[ci] = []
            metadata[ci] = []
            for aa in row["actors"][ci]["actors"]:
                actor = next(
                    v for v in frame["actors"] if v["actor_id"] == aa["actor_id"]
                )
                ref = actor_reference(actor, pose, T, K)
                reference_audit.append(
                    dict(
                        sample_id=row["sample_id"],
                        camera=cam,
                        actor_id=aa["actor_id"],
                        **ref,
                    )
                )
                accepted = [v for v in aa["identity_candidates"] if v["accepted"]]
                ids = [v["render_id"] for v in accepted]
                m = (
                    np.isin(idsimg, ids)
                    & np.isfinite(d["depth"][ci])
                    & (d["depth"][ci] >= 1.5)
                    & (d["depth"][ci] < 80)
                )
                if m.sum() < 3:
                    unknown_association += 1
                    continue
                vv, uu = np.where(m)
                pts = transform_points(
                    unproject_pixels_to_optical(uu, vv, d["depth"][ci, vv, uu], K), T
                )
                bottom = vv >= vv.max() - 1
                projected = transform_points(
                    np.array(actor["corners_hs_ego_m"]), np.linalg.inv(T)
                )
                front = projected[:, 2] > 0
                uvp = (K @ projected[front].T).T
                uvp = uvp[:, :2] / uvp[:, 2:]
                area = (
                    (np.ptp(uvp[:, 0]) + 1) * (np.ptp(uvp[:, 1]) + 1) if len(uvp) else 0
                )
                ref.update(
                    visible_surface_center_ego=pts.mean(0).tolist(),
                    visible_bottom_ego=pts[bottom].mean(0).tolist(),
                    native_visible_pixels=int(m.sum()),
                    visible_surface_not_box=True,
                )
                rec = dict(
                    split=a.split,
                    sample_id=row["sample_id"],
                    group=row["group"],
                    camera=cam,
                    camera_index=ci,
                    actor_id=aa["actor_id"],
                    type=(
                        "vehicle"
                        if aa["type_id"].startswith("vehicle.")
                        else "person" if aa["type_id"].startswith("walker.") else "cone"
                    ),
                    type_id=aa["type_id"],
                    pitch_deg=pitch,
                    configuration=row["configuration"],
                    timestamp=frame["timestamp_s"],
                    truncated=bool(
                        uu.min() == 0
                        or uu.max() == 639
                        or vv.min() == 0
                        or vv.max() == 383
                    ),
                    visibility_ratio_proxy=float(m.sum() / max(area, 1)),
                    association_purity=min(v["purity"] for v in accepted),
                    **ref,
                )
                truth[ci].append(m)
                refs[ci].append(rec)
                metadata[ci].append(aa)
        for seed, model in mobile.items():
            ti = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.fuse(model.levels(x["images"]))
                logits = model.head(features, (384, 640)).float().sigmoid()
            q = (
                (torch.clamp((logits * 256).floor(), max=255) / 256)
                .reshape(6, 384, 640)
                .cpu()
                .numpy()
            )
            torch.cuda.synchronize()
            timings.append(
                dict(
                    sample_id=row["sample_id"],
                    seed=seed,
                    mobile_ms=(time.perf_counter() - ti) * 1000,
                )
            )
            for ci, cam in enumerate(row["camera_names"]):
                K = np.array(inp["intrinsics"][ci])
                T = np.array(inp["cam_to_ego"][ci])
                ccs = components(q[ci])
                pairs, fp, fn = match(ccs, truth[ci])
                bycand = {i: (j, iou) for i, j, iou in pairs}
                byactor = {j: i for i, j, _ in pairs}
                overlap = np.array(
                    [[int((c["mask"] & m).sum()) for m in truth[ci]] for c in ccs],
                    dtype=int,
                ).reshape(len(ccs), len(truth[ci]))
                for j, ref in enumerate(refs[ci]):
                    actors.append(
                        dict(
                            ref,
                            seed=seed,
                            detected=j in byactor,
                            candidate_id=(
                                None
                                if j not in byactor
                                else f"{row['sample_id']}:{seed}:{cam}:{byactor[j]}"
                            ),
                            split_components=(
                                int((overlap[:, j] >= 3).sum()) if len(ccs) else 0
                            ),
                        )
                    )
                for i, c in enumerate(ccs):
                    idx = len(X)
                    key = f"{row['sample_id']}:{seed}:{cam}:{i}"
                    f = feature_vector(features[ci : ci + 1], c, q[ci], K, T)
                    X.append(f.cpu())
                    point, reason = locate(c, K, T)
                    vv, uu = np.where(c["mask"])
                    selection = np.linspace(0, len(uu) - 1, min(64, len(uu))).astype(
                        int
                    )
                    uv = np.stack([uu[selection], vv[selection]], -1)
                    pr = sample_map(
                        dense["probs"][0, ci : ci + 1],
                        torch.tensor(uv[None], device="cuda", dtype=torch.float32),
                        stride=8,
                        center=4,
                    )[0].mean(0)
                    pr /= pr.sum()
                    center_uv = np.array([uu.mean(), vv.mean()])
                    ray = T[:3, :3] @ np.linalg.solve(K, np.r_[center_uv, 1])
                    exp = float((pr * net.splat.depth_centers).sum())
                    mode = float(net.splat.depth_centers[pr.argmax()])
                    merge = int((overlap[i] >= 3).sum())
                    rec = dict(
                        candidate_id=key,
                        feature_index=idx,
                        seed=seed,
                        split=a.split,
                        sample_id=row["sample_id"],
                        group=row["group"],
                        camera=cam,
                        region=c["bbox"],
                        K=K.tolist(),
                        T=T.tolist(),
                        timestamp=frame["timestamp_s"],
                        q_mean=float(q[ci][c["mask"]].mean()),
                        pixel_area=int(c["mask"].sum()),
                        plane=None if point is None else point.tolist(),
                        plane_reason=reason,
                        dense_expected=(T[:3, 3] + exp * ray).tolist(),
                        dense_mode=(T[:3, 3] + mode * ray).tolist(),
                        foreground_center_uv=center_uv.tolist(),
                    )
                    candidates.append(rec)
                    if i not in bycand:
                        labels.append(
                            dict(
                                candidate_id=key,
                                feature_index=idx,
                                matched=False,
                                train_eligible=False,
                                reason="unmatched_unknown_not_confirmed_false_object",
                            )
                        )
                        continue
                    j, iou = bycand[i]
                    ref = refs[ci][j]
                    bx, by, bw, bh = c["bbox"]
                    offset = (
                        np.array(ref["projected_center_uv"])
                        - [bx + (bw - 1) / 2, by + (bh - 1) / 2]
                    ) / [bw, bh]
                    splitnum = int((overlap[:, j] >= 3).sum())
                    eligible = (
                        merge == 1
                        and splitnum == 1
                        and 1 <= ref["optical_z"] <= 80
                        and abs(offset).max() <= 2
                    )
                    labels.append(
                        dict(
                            candidate_id=key,
                            feature_index=idx,
                            matched=True,
                            train_eligible=bool(eligible),
                            reason=(
                                "unique" if eligible else "ambiguous_or_unrepresentable"
                            ),
                            merged_actor_count=merge,
                            split_components=splitnum,
                            iou=iou,
                            target=[
                                float(np.log(max(ref["optical_z"], 1e-6))),
                                *offset.tolist(),
                            ],
                            reference=ref,
                        )
                    )
        frame_audit.append(
            dict(
                sample_id=row["sample_id"],
                group=row["group"],
                synchronized=True,
                unassociated_actor_camera_entries=unknown_association,
                elapsed_s=time.perf_counter() - t,
            )
        )
        print(a.split, row["sample_id"], len(X), flush=True)
    features = torch.stack(X) if X else torch.empty((0, 600))
    torch.save(dict(rgb_features=features), dest / "features.pt")
    write(dest / "candidates.json", candidates)
    write(
        dest / "supervision.json",
        dict(
            actors=actors, labels=labels, references=reference_audit, frames=frame_audit
        ),
    )
    write(
        f"reports/m5n_cache_{a.split}.json",
        dict(
            kind="actual_RTX4090_frozen_RGB_features_and_separate_supervision",
            frames=len(frame_audit),
            groups=len({r["group"] for r in frame_audit}),
            candidates=len(candidates),
            visible_actor_views=len(actors) // 2,
            train_eligible={
                s: sum(
                    l["train_eligible"] and candidates[l["feature_index"]]["seed"] == s
                    for l in labels
                )
                for s in [101, 102]
            },
            features=asset(dest / "features.pt"),
            inference_candidates=asset(dest / "candidates.json"),
            supervision=asset(dest / "supervision.json"),
            elapsed_s=time.perf_counter() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            timings=timings,
            source_sha256=file_hash(__file__),
            limits="Simulator OBB center not contactGold. Native render identity geometrically associated, not exact mesh identity. Unmatched candidates not guaranteed falseobjects. Static historical snapshots read after same tick before nexttick, not all historical manifests carry peractor frame stamps.",
        ),
    )


if __name__ == "__main__":
    main()
