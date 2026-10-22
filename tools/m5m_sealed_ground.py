"""Sealed fixed/oracle plane and bottom-factor diagnostics, no threshold selection."""

import json
from pathlib import Path
from tools.m5l_common import write, summary


def main():
    rr = json.loads(Path("runs/m5m_contact_sealed.json").read_text())["actors"]
    out = []
    for seed in [101, 102]:
        for scope in ["all", "vehicle", "person", "cone", "flat", "slope"]:
            r = [
                r
                for r in rr
                if r["seed"] == seed
                and (
                    scope == "all"
                    or r["type"] == scope
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
            matched = [r for r in r if all(k in r for k in keys)]
            out.append(
                dict(
                    seed=seed,
                    scope=scope,
                    n=len(matched),
                    xy={k: summary([r[k]["xy_m"] for r in matched]) for k in keys},
                )
            )
    write(
        "reports/m5m_sealed_ground.json",
        dict(
            kind="offline_frozen_sealed_renderer_proxy_diagnostics",
            records=out,
            limits="No Gold ground or contact. All interventions matched same targets; no test selection. RGB local plane unpromoted, old-data diagnostic only.",
        ),
    )


if __name__ == "__main__":
    main()
