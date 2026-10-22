"""Lock method, selected weights and calibration before new sealed allocation."""

import json, subprocess
from pathlib import Path
from tools.m5n_common import write, asset
from simulator.recorder import file_hash


def main():
    for p in [
        "reports/m5n_validation_lock.json",
        "reports/m5n_actor_center_reference.json",
        "reports/m5n_eval_regression.json",
    ]:
        assert Path(p).exists()
    cfg = json.loads(Path("configs/m5j_models.json").read_text())
    models = {r["checkpoint"]: r["sha256"] for r in [cfg["depth"], *cfg["pixel"]]}
    models.update(
        {
            r["selected"]["path"]: r["selected"]["sha256"]
            for r in json.loads(
                Path("reports/m5n_direct_range_training.json").read_text()
            )["records"]
        }
    )
    sources = {
        str(p): file_hash(p)
        for root in ["tools", "models", "datasets", "diagnostics", "configs", "tests"]
        for p in Path(root).glob("*m5n*")
        if p.is_file()
    }
    write(
        "reports/m5n_pretest_freeze.json",
        dict(
            base=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            sources=sources,
            models=models,
            selection=asset("reports/m5n_validation_lock.json"),
            further_training=False,
            kind="before_new_test_allocation_and_capture",
        ),
    )


if __name__ == "__main__":
    main()
