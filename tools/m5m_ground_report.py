"""Separate plane error from bottom-pixel error with matched diagnostic interventions."""

import json
from pathlib import Path
from tools.m5l_common import write, asset, summary


def main():
    out = []
    planes = []
    for split in ["val", "regression", "test"]:
        records = json.loads(Path(f"runs/m5m_contact_{split}.json").read_text())[
            "actors"
        ]
        p = Path(f"runs/m5m_gating_{split}/raw.json")
        if p.exists():
            planes.extend(
                dict(r, split=split) for r in json.loads(p.read_text())["planes"]
            )
        for seed in [101, 102]:
            for scope in ["all", "flat", "slope", "vehicle", "person", "cone"]:
                rr = [
                    r
                    for r in records
                    if r["seed"] == seed
                    and (
                        scope == "all"
                        or scope == r["type"]
                        or scope == "flat"
                        and abs(r["ego_pitch_deg"]) <= 1
                        or scope == "slope"
                        and abs(r["ego_pitch_deg"]) > 1
                    )
                ]
                keys = [
                    "plane_bottom",
                    "oracle_plane_predbottom",
                    "fixed_plane_gtbottom",
                    "oracle_plane_gtbottom",
                ]
                same = [r for r in rr if all(k in r for k in keys)]
                out.append(
                    dict(
                        split=split,
                        seed=seed,
                        scope=scope,
                        n=len(same),
                        errors={k: summary([r[k]["xy_m"] for r in same]) for k in keys},
                        fixed_plane_height=summary(
                            [
                                r["fixed_plane_height_error_m"]
                                for r in rr
                                if r["fixed_plane_height_error_m"] is not None
                            ]
                        ),
                    )
                )
    write(
        "reports/m5m_ground_plane_diagnostics.json",
        dict(
            kind="offline_actual_renderer_and_RGB_depth_diagnostic",
            matched_interventions=out,
            rgb_plane=[
                dict(
                    split=s,
                    seed=seed,
                    detected_candidates=sum(
                        r["split"] == s and r["seed"] == seed for r in planes
                    ),
                    accepted=sum(
                        r["split"] == s and r["seed"] == seed and r["accepted"]
                        for r in planes
                    ),
                    xy=summary(
                        [
                            r["error"]["xy_m"]
                            for r in planes
                            if r["split"] == s and r["seed"] == seed and r["accepted"]
                        ]
                    ),
                )
                for s in ["val", "regression", "test"]
                for seed in [101, 102]
            ],
            limits="Oracle local road is rendered least-squares surface, not independent Gold ground. Bottom endpoint is visible native surface, not contact. Both GT interventions diagnostic-only. RGB plane uses unchanged predicted depth below RGB component excluding foreground, no GT road/map; not reliable road normal, not promoted. No new plane network.",
        ),
    )


if __name__ == "__main__":
    main()
