"""Separate footprint sampling diagnosis; explicitly not exact swept volume."""

import argparse
from pathlib import Path
import numpy as np
from simulator.oracle_driver import (
    OracleState,
    Trajectory,
    safety_check as baseline,
    footprint,
    polygons_intersect,
)
from simulator.motion_oracle import safety_check as improved
from simulator.recorder import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m46_safety_sampling.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    state = OracleState(0, 0, 0, 0, 1)
    tr = Trajectory(np.zeros((2, 4)), 1.0, 0.5)
    rows = []
    for speed in [5, 10, 20, 40, 80, 160]:
        for crossing_time in [0.1, 0.225, 0.25, 0.475]:
            obs = dict(
                xy=[0, -speed * crossing_time],
                yaw=0.0,
                half_length=0.1,
                half_width=0.1,
                velocity=[0, speed],
            )
            reference = any(
                polygons_intersect(
                    footprint(0, 0, 0, 2.6, 1.15),
                    footprint(0, -speed * crossing_time + speed * t, 0, 0.1, 0.1),
                )
                for t in np.arange(0, 1, 0.001)
            )
            rows.append(
                dict(
                    speed_mps=speed,
                    crossing_time=crossing_time,
                    reference_1ms_collision=reference,
                    baseline_detects=baseline(tr, state, [obs]) is not None,
                    m46_detects=improved(tr, state, [obs]) is not None,
                )
            )
    atomic_json(
        a.output,
        dict(
            schema_version="hsd-m46-safety-sampling-v1",
            rows=rows,
            baseline_misses=sum(
                r["reference_1ms_collision"] and not r["baseline_detects"] for r in rows
            ),
            m46_misses=sum(
                r["reference_1ms_collision"] and not r["m46_detects"] for r in rows
            ),
            scope="Stationary ego, linear tiny crossing footprint; 1ms discrete reference also not exact continuous. High speeds outside operating range deliberately expose sampling limitation.",
            recommendation="Adaptive time step by relative translational/rotational swept distance or conservative swept polygon/continuous separating axis bound. Need rotation, road area and dynamic uncertainty coverage before full safety claim.",
        ),
    )


if __name__ == "__main__":
    main()
