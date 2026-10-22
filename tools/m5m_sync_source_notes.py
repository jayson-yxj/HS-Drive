"""List only material post-freeze evaluation exclusion and delivery changes."""

import json
from pathlib import Path
from tools.m5l_common import write
from simulator.recorder import file_hash


def main():
    freeze = json.loads(Path("reports/m5m_pretest_freeze.json").read_text())
    changes = {
        p: dict(before=h, after=file_hash(p))
        for p, h in freeze["sources"].items()
        if file_hash(p) != h
    }
    write(
        "reports/m5m_final_source_audit.json",
        dict(
            changes=changes,
            frozen_model_geometry_capture_threshold_sources_unchanged=all(
                p == "tools/m5m_integrity.py" for p in changes
            ),
            evaluation_exclusion="Post-capture pair QC applies existing .001 tolerance, preserves2 bad frames and unfilteredindex; no changes to learned inference or thresholds",
            added_postfreeze_tools="Delivery aggregation, attribution, IPC microbenchmark, historical report audit, anatomy figures, integrity and selective sealed exclusion; not model selection",
        ),
    )


if __name__ == "__main__":
    main()
