"""Separate loop cadence and cold/steady results from shared-GPU trial."""

import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary


def main():
    raw = json.loads(Path("runs/m5m_logs/shadow_raw.json").read_text())
    phases = []
    for r in raw["phases"]:
        vals = np.array(r["raw_interval_ms"])
        phases.append(
            dict(
                repeat=r["repeat"],
                phase=r["phase"],
                mean_hz=1000 / vals.mean() if len(vals) else None,
                interval_ms=summary(vals),
                deadline_over50ms=int((vals > 50).sum()),
                deadline_over55ms=int((vals > 55).sum()),
                n=len(vals),
                tick_ms=r["tick_ms"],
                enqueue_ms=r["queue_enqueue_ms"],
                drain_ms=r["output_drain_ms"],
            )
        )
    outputs = raw["outputs"]
    write(
        "reports/m5m_shadow_cadence.json",
        dict(
            kind="actual_CARLA_static_resource_trial_postprocess",
            phases=phases,
            cold=[r for r in outputs if r.get("cold_start")],
            worker_initialization=[
                r for r in outputs if r["status"] == "worker_initialized"
            ],
            steady=[
                dict(
                    repeat=rep,
                    **{
                        k: summary(
                            [
                                r[k]
                                for r in outputs
                                if r.get("repeat") == rep
                                and k in r
                                and not r.get("cold_start", False)
                            ]
                        )
                        for k in [
                            "inference_ms",
                            "end_to_end_ms",
                            "queue_age_ms",
                            "convert_h2d_ms",
                        ]
                    },
                )
                for rep in [0, 1]
            ],
            status={
                s: sum(r["status"] == s for r in outputs)
                for s in sorted({r["status"] for r in outputs})
            },
            limits="No driving. Report phase/setup separately, paired order insufficient for causal certainty. Shared GPU not hardware isolation. Raw control-loop cadence has no vehicle controller.",
        ),
    )


if __name__ == "__main__":
    main()
