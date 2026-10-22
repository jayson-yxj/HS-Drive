"""Immutable method/model/validation locks before sealed allocation or capture."""

import json, subprocess
from pathlib import Path
from tools.m5l_common import write, asset
from simulator.recorder import file_hash


def main():
    for p in [
        "reports/m5m_selective_lock.json",
        "reports/m5m_residual_lock.json",
        "reports/m5m_anatomy_dev.json",
        "reports/m5m_dev_execution.json",
    ]:
        assert Path(p).exists(), p
    sources = {
        str(p): file_hash(p)
        for root in ["tools", "models", "diagnostics", "simulator", "configs", "tests"]
        for p in Path(root).glob("*m5m*")
        if p.is_file()
    }
    cfg = json.loads(Path("configs/m5j_models.json").read_text())
    models = {r["checkpoint"]: r["sha256"] for r in [cfg["depth"], *cfg["pixel"]]}
    models.update(
        {
            r["selected"]["path"]: r["selected"]["sha256"]
            for r in json.loads(Path("reports/m5l_vehicle_training.json").read_text())[
                "records"
            ]
        }
    )
    for p, h in models.items():
        assert file_hash(p) == h, p
    write(
        "reports/m5m_pretest_freeze.json",
        dict(
            kind="before_sealed_geometry_allocation_capture",
            base=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            sources=sources,
            models=models,
            locks=[
                asset("reports/m5m_selective_lock.json"),
                asset("reports/m5m_residual_lock.json"),
            ],
            contact_head_trained=False,
            reason="Gold support not met; no additional training; historical tests regression only",
        ),
    )


if __name__ == "__main__":
    main()
