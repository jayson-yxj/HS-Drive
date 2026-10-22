import numpy as np
import pytest
from datasets.m5o_benchmark import (
    evaluate,
    unique_target_key,
    validate_split_isolation,
    support,
)
from tools.m5o_candidate_audit import category
from tools.m5o_cache import oracle_component


def actor(detected=True, key="c", frame="s"):
    return dict(
        sample_id=frame,
        camera="front",
        actor_id=7,
        group="g",
        detected=detected,
        candidate_id=key,
        box_center_ego=[10, 0, 0],
        type_id="vehicle.tesla.model3",
    )


def test_full_denominator_and_abstention():
    aa = [actor(), actor(False, frame="s2")]
    p = {"c": dict(positions={"direct": [13, 4, 0]}, confidence=0.3)}
    r = evaluate(aa, p, "direct")
    assert (
        r["coverage"] == 0.5 and r["xy"]["mean"] == 5 and r["xy"]["severe5m_rate"] == 0
    )
    r = evaluate(aa, p, "direct", 0.4)
    assert r["coverage"] == 0 and r["xy"]["mean"] is None and r["unlocalized"] == 2


def test_duplicate_actor_denominator_rejected():
    with pytest.raises(ValueError):
        evaluate([actor(), actor()], {}, "direct")


def test_unmatched_not_physical_fp():
    assert (
        category(dict(matched=False, semantic_histogram={"14": 99, "7": 1}))
        == "unrecorded_or_unassociated_target_surface"
    )
    assert (
        category(dict(matched=False, semantic_histogram={"3": 100}))
        == "rendered_background_semantic_proxy"
    )


def test_cross_split_road_and_distance():
    a = dict(group="a", split="train", map="town", road_id=1, pose=[0, 0])
    b = dict(group="b", split="val", map="town", road_id=1, pose=[1000, 0])
    assert validate_split_isolation([a, b])
    b["road_id"] = 2
    assert not validate_split_isolation([a, b])
    b["pose"] = [100, 0]
    assert validate_split_isolation([a, b])


def test_unique_target_not_frame_or_actor_id():
    a = actor()
    b = dict(a, sample_id="other", actor_id=99)
    assert unique_target_key(a) == unique_target_key(b)
    b["camera"] = "rear"
    assert unique_target_key(a) != unique_target_key(b)


def test_oracle_mask_preserves_disconnected_instance():
    m = np.zeros((20, 30), bool)
    m[4:6, 3:5] = True
    m[14:16, 20:22] = True
    c = oracle_component(m)
    assert c["bbox"] == [3, 4, 19, 12] and c["area"] == 8


def test_unchanged_rgb_whitelist():
    from datasets.inputs import INPUT_FIELDS

    assert not any(
        "depth" in k or "actor" in k or "instance" in k for k in INPUT_FIELDS
    )


def test_shadow_time_and_control_domains():
    from datasets.m5o_interfaces import SpatialEnvelope, ObjectEstimate

    envelope = SpatialEnvelope("m5o-object-v1", "rgb_object_candidates", 1, 2, 2.2)
    envelope.validate(2.1)
    for now in [1.9, 2.3, float("nan")]:
        with pytest.raises(ValueError):
            envelope.validate(now)
    with pytest.raises(ValueError):
        envelope.validate(2.1, consumer="controller")
    ObjectEstimate("c", "DetectedNoLocation", None, None).validate()
    with pytest.raises(ValueError):
        ObjectEstimate("c", "DetectedNoLocation", (1, 2, 3), 1).validate()
    with pytest.raises(ValueError):
        ObjectEstimate(
            "c", "DetectedWithEstimate", (1, 2, 3), 1, safety_free=True
        ).validate()


def test_risk_features_ignore_gt_fields():
    from diagnostics.m5n_metrics import risk_features

    c = dict(region=[10, 20, 30, 40], pixel_area=400, q_mean=0.95)
    a = risk_features(c, [10, 2, 0])
    c.update(box_center_ego=[100, 100, 0], error=999, depth_gt=123)
    np.testing.assert_array_equal(a, risk_features(c, [10, 2, 0]))


def test_calibration_folds_are_location_disjoint():
    from diagnostics.m5o_risk import fit_calibration

    cs = []
    aa = []
    pp = {}
    for g in range(4):
        for i in range(6):
            key = f"{g}_{i}"
            cs.append(
                dict(
                    candidate_id=key,
                    group=str(g),
                    region=[10 + i, 20, 30, 40],
                    pixel_area=400,
                    q_mean=0.95,
                )
            )
            a = actor(key=key, frame=key)
            a["group"] = str(g)
            aa.append(a)
            pp[key] = dict(positions={"direct": [11 + g + i / 5, 0, 0]})
    cal, oof = fit_calibration(cs, aa, pp)
    assert len(oof) == 24
    for fold in cal["folds"]:
        assert fold["heldout"] not in fold["train_groups"]


def test_fail_closed_quality_gate():
    from diagnostics.m5o_risk import quality_gate
    from datasets.m5o_benchmark import METHODS

    a = actor()
    p = {"c": dict(positions={m: [10, 0, 0] for m in METHODS}, confidence=1)}
    assert not quality_gate([a], p, 0)["passed"]
    assert not quality_gate([a], p, 1.1)["passed"]


def test_ambiguous_detection_not_localization_success():
    a = actor()
    a["unique_association"] = False
    p = {"c": dict(positions={"direct": [10, 0, 0]}, confidence=1)}
    r = evaluate([a], p, "direct")
    assert r["detected"] == 1 and r["localized"] == 0 and r["ambiguous_matched"] == 1


def test_empty_risk_report_not_nan():
    from diagnostics.m5o_risk import risk_report
    r=risk_report([],{});assert r['n']==0 and r['group_p95_bootstrap95'] is None


def test_unique_support_repeated_frame_not_new_sample():
    c=[dict(seed=101,feature_index=i,candidate_id=str(i),group='g',camera='front') for i in range(2)]
    ref=dict(actor(),type='vehicle',ego_xy_range_m=10)
    labels=[dict(feature_index=i,train_eligible=True,reference=dict(ref,sample_id=str(i))) for i in range(2)]
    r=support(c,labels,101);assert r['unique_candidates']==1
