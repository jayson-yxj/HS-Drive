"""Unified object-view accounting; detector, association, localization are separate."""

from collections import Counter
import numpy as np

METHODS = ("plane", "dense_expected", "dense_mode", "geometry_prior", "direct")


def unique_target_key(ref):
    # Actor IDs are world-local, not stable identity. Static duplicate frames collapse.
    xyz = tuple(np.round(ref["box_center_ego"], 2))
    return (ref["group"], ref["camera"], ref["type_id"], xyz)


def support(candidates, labels, seed):
    seen = set()
    records = []
    for label in labels:
        c = candidates[label["feature_index"]]
        if c["seed"] != seed or not label["train_eligible"]:
            continue
        ref = label["reference"]
        key = unique_target_key(ref)
        if key in seen:
            continue
        seen.add(key)
        records.append((c, ref))
    return dict(
        unique_candidates=len(records),
        groups=len({c["group"] for c, r in records}),
        by_class=dict(Counter(r["type"] for c, r in records)),
        class_locations={
            k: len({c["group"] for c, r in records if r["type"] == k})
            for k in ["vehicle", "person", "cone"]
        },
        camera_locations={
            k: len({c["group"] for c, r in records if c["camera"] == k})
            for k in [
                "front",
                "front_left",
                "front_right",
                "rear",
                "rear_left",
                "rear_right",
            ]
        },
        by_camera=dict(Counter(c["camera"] for c, r in records)),
        by_distance=dict(
            Counter(
                (
                    "0-10"
                    if r["ego_xy_range_m"] < 10
                    else (
                        "10-20"
                        if r["ego_xy_range_m"] < 20
                        else "20-40" if r["ego_xy_range_m"] < 40 else "40-80"
                    )
                )
                for c, r in records
            )
        ),
    )


def quality(candidates, supervision, seed):
    cc = [c for c in candidates if c["seed"] == seed]
    ids = {c["candidate_id"] for c in cc}
    ll = [l for l in supervision["labels"] if l["candidate_id"] in ids]
    aa = [a for a in supervision["actors"] if a["seed"] == seed]
    return dict(
        seed=seed,
        candidates=len(cc),
        matched=sum(l["matched"] for l in ll),
        unmatched_unknown=sum(not l["matched"] for l in ll),
        physical_false_positive=None,
        visible_actor_views=len(aa),
        detected=sum(a["detected"] for a in aa),
        missed=sum(not a["detected"] for a in aa),
        split_actor_views=sum(a["split_components"] > 1 for a in aa),
        merged_candidates=sum(l.get("merged_actor_count", 0) > 1 for l in ll),
        unique_support=support(candidates, supervision["labels"], seed),
    )


def summary(values):
    x = np.asarray(values, float)
    if not len(x):
        return dict(
            n=0,
            mean=None,
            median=None,
            p90=None,
            p95=None,
            max=None,
            severe5m_rate=None,
        )
    return dict(
        n=len(x),
        mean=float(x.mean()),
        median=float(np.median(x)),
        p90=float(np.quantile(x, 0.9)),
        p95=float(np.quantile(x, 0.95)),
        max=float(x.max()),
        severe5m_rate=float((x > 5).mean()),
    )


def evaluate(actors, predictions, method, threshold=0):
    if len({(a["sample_id"], a["camera"], a["actor_id"]) for a in actors}) != len(
        actors
    ):
        raise ValueError("one seed, unique actor-views required")
    errors = []
    groups = []
    accepted = []
    for a in actors:
        if not a["detected"] or not a.get("unique_association", True):
            continue
        p = predictions[a["candidate_id"]]
        pos = p["positions"].get(method)
        if pos is None or not np.isfinite(pos).all():
            continue
        if p.get("confidence", 1) < threshold:
            continue
        errors.append(
            float(np.linalg.norm(np.array(pos)[:2] - np.array(a["box_center_ego"])[:2]))
        )
        groups.append(a["group"])
        accepted.append(a["candidate_id"])
    return dict(
        visible_actor_views=len(actors),
        detected=sum(a["detected"] for a in actors),
        detection_rate=sum(a["detected"] for a in actors) / max(len(actors), 1),
        ambiguous_matched=sum(
            a["detected"] and not a.get("unique_association", True) for a in actors
        ),
        localized=len(errors),
        coverage=len(errors) / max(len(actors), 1),
        groups=len(set(groups)),
        xy=summary(errors),
        accepted_ids=accepted,
        unlocalized=len(actors) - len(errors),
    )


def validate_split_isolation(groups, min_cross_m=150):
    bad = []
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            if a["group"] == b["group"]:
                bad.append([a["group"], b["group"], "duplicate"])
            if a["split"] == b["split"] or a["map"] != b["map"]:
                continue
            if (
                a["road_id"] == b["road_id"]
                or np.linalg.norm(np.array(a["pose"][:2]) - b["pose"][:2]) < min_cross_m
            ):
                bad.append([a["group"], b["group"], "cross_split_road_or_distance"])
    return bad
