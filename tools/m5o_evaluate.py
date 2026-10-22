"""Evaluate frozen models; Calibration independent of Validation; sealed read once."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from tools.m5o_train import read
from tools.m5o_common import write, asset, provenance
from models.m5n_direct import DirectCenterHead, reconstruct
from datasets.m5o_benchmark import evaluate, METHODS
from diagnostics.m5o_risk import (
    fit_calibration,
    apply_calibration,
    quality_gate,
    risk_report,
)
from training.runtime import prepare_cuda
from simulator.recorder import file_hash


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "calibration", "sealed"], required=True)
    a = p.parse_args()
    out = Path(f"runs/m5o_eval_{a.split}")
    out.mkdir(exist_ok=False)
    if a.split == "sealed":
        lock = json.loads(Path("reports/m5o_pretest_freeze.json").read_text())
        for p, h in lock["assets"].items():
            assert file_hash(p) == h, p
    prepare_cuda()
    torch.set_num_threads(2)
    X, cc, sup = read(a.split)
    training = json.loads(Path("reports/m5o_training.json").read_text())
    results = {}
    records = []
    strata = []
    calibrations = {}
    risks = []
    curves = []
    burden = []
    for fraction in [0.25, 0.5, 1.0] if a.split == "val" else [1.0]:
        for seed in [101, 102]:
            cs = [c for c in cc if c["seed"] == seed]
            feats = X[[c["feature_index"] for c in cs]].cuda()
            regions = torch.tensor([c["region"] for c in cs], device="cuda")
            K = torch.tensor([c["K"] for c in cs], device="cuda")
            T = torch.tensor([c["T"] for c in cs], device="cuda")
            preds = {
                c["candidate_id"]: dict(
                    candidate_id=c["candidate_id"],
                    group=c["group"],
                    camera=c["camera"],
                    timestamp=c["timestamp"],
                    positions={m: c[m] for m in METHODS[:3]},
                    confidence=1,
                )
                for c in cs
            }
            for arm in ["direct", "geometry_prior"]:
                rec = next(
                    r
                    for r in training["records"]
                    if r["seed"] == seed
                    and r["arm"] == arm
                    and r["fraction"] == fraction
                )["selected"]
                assert file_hash(rec["path"]) == rec["sha256"]
                ck = torch.load(rec["path"], weights_only=True)
                model = DirectCenterHead(ck["input_dim"]).cuda().eval()
                model.load_state_dict(ck["model"])
                xyz = (
                    reconstruct(
                        model(feats if arm == "direct" else feats[:, -24:]),
                        regions,
                        K,
                        T,
                    )
                    .cpu()
                    .numpy()
                )
                for c, point in zip(cs, xyz):
                    preds[c["candidate_id"]]["positions"][arm] = point.tolist()
            actors = [v for v in sup["actors"] if v["seed"] == seed]
            threshold = 0
            if a.split == "calibration":
                cal, oof = fit_calibration(cs, actors, preds)
                apply_calibration(cal, cs, preds, oof)
                calibrations[str(seed)] = cal
                local = [
                    quality_gate(actors, preds, t) for t in [0, 0.2, 0.4, 0.6, 0.8]
                ]
                passed = [r for r in local if r["passed"]]
                threshold = min([r["threshold"] for r in passed]) if passed else 1.1
                cal["threshold"] = threshold
                cal["promoted"] = bool(passed)
                curves.extend(dict(seed=seed, **r) for r in local)
            elif a.split == "sealed":
                cal = json.loads(Path("reports/m5o_calibration_lock.json").read_text())[
                    "calibration"
                ][str(seed)]
                apply_calibration(cal, cs, preds)
                threshold = cal["threshold"]
                curves.append(dict(seed=seed, **quality_gate(actors, preds, threshold)))
            if a.split != "val":
                risks.append(dict(seed=seed, **risk_report(actors, preds)))
            # Natural coverage comparison; selected threshold is additional separate result.
            for m in METHODS:
                records.append(
                    dict(
                        seed=seed,
                        fraction=fraction,
                        method=m,
                        **evaluate(actors, preds, m, 0),
                    )
                )
            if a.split != "val":
                records.append(
                    dict(
                        seed=seed,
                        fraction=fraction,
                        method="direct_selected",
                        **evaluate(actors, preds, "direct", threshold),
                    )
                )
            fields = {
                "type": lambda x: x["type"],
                "camera": lambda x: x["camera"],
                "distance": lambda x: (
                    "0-10"
                    if x["ego_xy_range_m"] < 10
                    else (
                        "10-20"
                        if x["ego_xy_range_m"] < 20
                        else "20-40" if x["ego_xy_range_m"] < 40 else "40-80"
                    )
                ),
                "group": lambda x: x["group"],
                "geometry": lambda x: "slope" if abs(x["pitch_deg"]) > 1 else "flat",
                "visibility": lambda x: (
                    "truncated"
                    if x["truncated"]
                    else (
                        "low_proxy_visibility"
                        if x["visibility_ratio_proxy"] < 0.3
                        else "visible"
                    )
                ),
            }
            if fraction == 1:
                for field, fn in fields.items():
                    for value in sorted({fn(x) for x in actors}):
                        for m in METHODS:
                            strata.append(
                                dict(
                                    seed=seed,
                                    field=field,
                                    value=value,
                                    method=m,
                                    **evaluate(
                                        [x for x in actors if fn(x) == value], preds, m
                                    ),
                                )
                            )
            matched = {x["candidate_id"] for x in actors if x["detected"]}
            burden.append(
                dict(
                    seed=seed,
                    fraction=fraction,
                    total_candidates=len(cs),
                    unmatched=len(cs) - len(matched),
                    accepted_all_candidates=sum(
                        p["confidence"] >= threshold for p in preds.values()
                    ),
                    accepted_unmatched_unknown=sum(
                        p["confidence"] >= threshold and k not in matched
                        for k, p in preds.items()
                    ),
                    threshold=threshold,
                    physical_fp=None,
                )
            )
            for key, pred in preds.items():
                accepted = pred["confidence"] >= threshold
                pred["output"] = dict(
                    candidate_id=key,
                    reference="ActorBoxCenter_research_not_contact",
                    status="DetectedWithEstimate" if accepted else "DetectedNoLocation",
                    position_ego_xyz_m=(
                        pred["positions"]["direct"] if accepted else None
                    ),
                    timestamp_s=pred["timestamp"],
                    confidence=pred["confidence"],
                    control_authorized=False,
                    safety_free=False,
                )
            results[f"{seed}_{fraction}"] = preds
    write(out / "predictions.json", results)
    if a.split == "calibration":
        write(
            "reports/m5o_calibration_lock.json",
            dict(
                calibration=calibrations,
                curves=curves,
                provenance=provenance(),
                model_selection="Validation only; independent Calibration fits risk and thresholds",
            ),
        )
    write(
        f"reports/m5o_eval_{a.split}.json",
        dict(
            kind="actual_GPU_model_and_offline_metrics",
            records=records,
            strata=strata,
            risks=risks,
            curves=curves,
            all_candidate_burden=burden,
            predictions=asset(out / "predictions.json"),
            provenance=provenance(),
        ),
    )


if __name__ == "__main__":
    main()
