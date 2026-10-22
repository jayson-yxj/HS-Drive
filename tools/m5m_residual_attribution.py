"""Partition all reliable rays: associated actor, nonassociated foreground, background."""

import json
from pathlib import Path
from tools.m5l_common import write, asset


def main():
    results = []
    for split in ["val", "regression", "test"]:
        d = json.loads(Path(f"reports/m5m_gating_{split}.json").read_text())
        ix = {(r["method"], r["scope"], r["band"]): r for r in d["depth"]}
        for r in d["depth"]:
            if r["scope"] != "all":
                continue
            b = ix.get((r["method"], "background", r["band"]), {"n": 0})
            a = ix.get((r["method"], "actor", r["band"]), {"n": 0})
            n = r["n"] - b["n"] - a["n"]
            if n:
                fields = {
                    k: (r[k] * r["n"] - b.get(k, 0) * b["n"] - a.get(k, 0) * a["n"]) / n
                    for k in ["expected_mae", "bias", "space1", "gate"]
                }
                results.append(
                    dict(
                        split=split,
                        method=r["method"],
                        band=r["band"],
                        scope="unassociated_foreground",
                        n=n,
                        **fields,
                    )
                )
    write(
        "reports/m5m_residual_attribution.json",
        dict(
            kind="offline_exact_partition_of_actual_GPU_sums",
            records=results,
            partition="all reliable GT-in-BEV = associated_actor + unassociated_foreground + reliable_background; Unknown separate and not truth",
            finding="Validation35–80m has783 reliable rays:575 background,208 unassociated foreground,0 associated Actor. RGB gating preserves most reliable background; far all-surface degradation dominated by unassociated foreground. Cannot claim these208 are vehicle/ped/cone without reliable identity. Global/per-position far protection still fails. No further training authorized by gate.",
            limits="Derived weighted sums may differ at floating rounding; nonassociated foreground includes rendered targets without capturedActor assignment, not negative background or Unknown.",
        ),
    )


if __name__ == "__main__":
    main()
