"""Certified target-surface point presence/absence; not whole-cell empty volume."""

import numpy as np
import cv2
from datasets.m5b_pairs import camera_rays, pair_witness
from geometry.ray_tracing import VoxelGrid


def certified_witness(root, man, a, b, la, lb, cache=None):
    cache = {} if cache is None else cache

    def get(f, name):
        key = (f["frame_id"], name)
        if key not in cache:
            cache[key] = camera_rays(root, man, f, name)
        return cache[key]

    legacy, details = pair_witness(root, man, a, b, la, lb, cache)
    grid = VoxelGrid()
    size = int(np.prod(grid.shape))
    counts = [np.zeros(size, np.int32) for _ in range(2)]
    denseobs = [np.zeros(size, bool) for _ in range(2)]
    diagnostics = []
    for name in a["cameras"]:
        ra, da = get(a, name)
        rb, db = get(b, name)
        for side, (frame, r, other, rd, od, own, opp) in enumerate(
            [(a, ra, rb, da, db, la, lb), (b, rb, ra, db, da, lb, la)]
        ):
            ids, inside = grid.locate(r["endpoints"])
            seen = inside & r["reliable"] & np.isin(r["semantics"], [2, 3])
            denseobs[side][ids[seen]] = True
            validmap = np.zeros(od[0].shape, bool)
            ou, ov = other["uv"].T
            validmap[ov, ou] = other["reliable"] & np.isin(
                other["semantics"], [1, 2, 3, 4]
            )
            validmap &= (
                other["stable_mask"]
                & ~other["self_mask"]
                & np.isfinite(od[0])
                & (od[0] >= 1.5)
                & (od[0] < 80)
            )
            u, v = r["uv"].T
            far = cv2.erode(od[0], np.ones((3, 3), np.uint8))
            candidate = np.zeros(len(ids), bool)
            for actor in frame["actors"]:
                if actor["is_ego"]:
                    continue
                box = np.asarray(actor["corners_hs_ego_m"])
                candidate |= (
                    (r["endpoints"] >= box.min(0) - 0.05)
                    & (r["endpoints"] <= box.max(0) + 0.05)
                ).all(-1)
            keep = (
                inside
                & r["reliable"]
                & np.isin(r["semantics"], [2, 3])
                & candidate
                & validmap[v, u]
                & (far[v, u] > rd[0][v, u] + 0.5)
                & (rd[2][v, u] != od[2][v, u])
            )
            ids = ids[keep]
            safe = np.ones(len(ids), bool)
            for lab in [own, opp]:
                for k in [
                    "ego_mask",
                    "evidence_conflict_mask",
                    "semantic_conflict_mask",
                ]:
                    safe &= ~lab[k].ravel()[ids]
            safe &= ~opp["surface_semantic"][1].ravel()[ids]
            np.add.at(counts[side], ids[safe], 1)
            diagnostics.append(
                dict(camera=name, present_side=side, reliable_pixels=int(safe.sum()))
            )
    vanish = (counts[0] >= 3) & ~denseobs[1]
    appear = (counts[1] >= 3) & ~denseobs[0]
    conflict = vanish & appear
    vanish &= ~conflict
    appear &= ~conflict
    vanish = vanish.reshape(grid.shape) & legacy["vanished"]
    appear = appear.reshape(grid.shape) & legacy["appeared"]
    stable = legacy["stable"].copy()
    for lab in [la, lb]:
        for key in ["ego_mask", "evidence_conflict_mask", "semantic_conflict_mask"]:
            stable &= ~lab[key]
    return dict(
        appeared=appear,
        vanished=vanish,
        stable=stable,
        legacy_appeared=legacy["appeared"],
        legacy_vanished=legacy["vanished"],
        legacy_stable=legacy["stable"],
        witness_count_a=counts[0].reshape(grid.shape),
        witness_count_b=counts[1].reshape(grid.shape),
    ), dict(
        pixels=diagnostics,
        legacy=details,
        retained=int(appear.sum() + vanish.sum()),
        legacy_count=int(legacy["appeared"].sum() + legacy["vanished"].sum()),
    )
