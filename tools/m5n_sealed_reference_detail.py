"""Check newly explicit bbox fields against recovered saved vertex geometry."""

import json
from pathlib import Path
import numpy as np
from tools.m5n_common import write, stats, rows
from tools.m5f_common import input_path
from datasets.m5n_reference import actor_reference


def main():
    records = []
    for row in rows("sealed"):
        f = json.loads(Path(row["raw_frame"]).read_text())
        inp = json.loads(input_path(row).read_text())
        for a in f["actors"]:
            if a["is_ego"]:
                continue
            r = actor_reference(
                a, inp["ego_pose"], inp["cam_to_ego"][0], inp["intrinsics"][0]
            )
            center = float(
                np.max(abs(np.array(r["bbox_local_center"]) - a["bbox_local_location"]))
            )
            dims = float(
                np.max(
                    abs(
                        np.array(r["bbox_dimensions_xyz"])
                        - 2 * np.array(a["bbox_extent"])
                    )
                )
            )
            pitch, yaw, roll = np.radians(a["bbox_rotation"])
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            cr, sr = np.cos(roll), np.sin(roll)
            R = np.array(
                [
                    [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                    [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                    [sp, -cp * sr, cp * cr],
                ]
            )
            rot = float(np.max(abs(R - np.array(r["bbox_local_rotation"]))))
            records.append(
                dict(
                    sample_id=row["sample_id"],
                    actor_id=a["actor_id"],
                    center_delta_m=center,
                    dimensions_delta_m=dims,
                    rotation_matrix_delta=rot,
                )
            )
    assert max(r["center_delta_m"] for r in records) < 0.001
    assert max(r["dimensions_delta_m"] for r in records) < 0.001
    assert max(r["rotation_matrix_delta"] for r in records) < 0.001
    write(
        "reports/m5n_sealed_box_definition_check.json",
        dict(
            kind="offline_actual_CARLA_explicit_bbox_vs_saved_vertices",
            records=len(records),
            center=stats([r["center_delta_m"] for r in records]),
            dimensions=stats([r["dimensions_delta_m"] for r in records]),
            rotation=stats([r["rotation_matrix_delta"] for r in records]),
            passed=True,
            limits="Simulation-defined bbox contract, not exact rendering Mesh; out-of-view hidden actors included in geometry consistency check.",
        ),
    )


if __name__ == "__main__":
    main()
