"""Recover M46 internal HOLD from saved inputs; offline replay, not measurement."""

import argparse
import json
from pathlib import Path

import numpy as np

from simulator.motion_oracle import MotionController, OracleState, Trajectory
from simulator.recorder import atomic_json, file_hash


def replay_hold(rows):
    """Replay the frozen controller's state transitions without touching CARLA.

    The initial warmup effort was not logged. HOLD decisions do not depend on
    effort; do not use this replay to infer actuator commands or vehicle motion.
    """
    controller = MotionController()
    active = []
    first_entry = None
    for row in rows:
        timestamp = row["measurement"]["t"]
        state = OracleState(row["x"], row["y"], row["yaw"], row["v"], timestamp)
        trajectory = None
        if row["trajectory"] is not None:
            trajectory = Trajectory(
                np.asarray(row["trajectory"], dtype=float), timestamp,
                row["trajectory_dt"],
            )
        was_holding = controller.holding
        controller.command(
            trajectory, state, row["safety_reason"],
            remaining_m=row["plan_remaining"],
        )
        active.append(controller.holding)
        if controller.holding and not was_holding and first_entry is None:
            first_entry = dict(
                t=row["t"], speed=row["v"], gap=row["gap"],
                position=[row["x"], row["y"]],
            )
    return active, first_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/m47_hold_replay.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    source = "simulator/motion_oracle.py"
    records = []
    for path in ["reports/m47_stopping_verified.json", "reports/m47_routes_verified.json"]:
        report = json.loads(Path(path).read_text())
        assert file_hash(source) == report["source_hashes_at_start"][source]
        for result in report["results"]:
            if result["version"] != "m46":
                continue
            assert file_hash(result["log"]) == result["log_sha256"]
            rows = [json.loads(line) for line in Path(result["log"]).read_text().splitlines()]
            active, entry = replay_hold(rows)
            records.append(dict(
                scenario=result["scenario"], log=result["log"],
                log_sha256=result["log_sha256"], first_entry=entry,
                active_ticks=sum(active),
            ))
    atomic_json(args.output, dict(
        kind="offline_frozen_controller_state_replay", records=records,
        controller_sha256=file_hash(source), source_sha256=file_hash(__file__),
        limitation="M46 internal HOLD was not logged. Recover transitions from recorded inputs; do not infer actuator commands from this replay.",
    ))


if __name__ == "__main__":
    main()
