"""Assemble final requested filenames; retain initial development reports as raw assets."""

import json, shutil
from pathlib import Path
from tools.m5l_common import asset


def update(name, changes):
    p = Path("reports") / (name + ".json")
    old = Path("runs/m5m_logs") / (name + "_development.json")
    if old.exists():
        raise FileExistsError(old)
    shutil.copyfile(p, old)
    d = json.loads(p.read_text())
    d.update(changes)
    d["development_report"] = asset(old)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    complete = json.loads(Path("reports/m5m_selective_complete.json").read_text())
    support = json.loads(Path("reports/m5m_contact_support_complete.json").read_text())
    update(
        "m5m_selective_localization",
        dict(
            aggregate=complete["aggregate"],
            strata=complete["strata"],
            candidate_counts=complete["detection"],
            complete=asset("reports/m5m_selective_complete.json"),
            limits=complete["limits"],
        ),
    )
    update(
        "m5m_risk_coverage",
        dict(
            curves=complete["risk_curves"],
            complete=asset("reports/m5m_selective_complete.json"),
            sealed_workpoint="Validation threshold0 retained diagnostic, not promoted; no sealed tuning",
        ),
    )
    update(
        "m5m_contact_reference_quality",
        dict(
            all_dataset_support=support["records"],
            new_anatomy=support["anatomy"],
            contact_head=support["contact_head"],
            fresh_capture_integrity=[
                asset(f"reports/m5m_capture_integrity_{p}.json")
                for p in ["dev", "sealed"]
            ],
        ),
    )
    update(
        "m5m_contact_visibility",
        dict(
            new_data=asset("reports/m5m_selective_complete.json"),
            new_anatomy=support["anatomy"],
            new_data_note="Use type/camera/visibility_proxy strata and detection counts; GT visibility proxies never inference features",
        ),
    )
    update("m5m_shadow_runtime", dict(cadence=asset("reports/m5m_shadow_cadence.json")))
    p = Path("reports/m5m_summary.md")
    p.write_text(
        p.read_text()
        + "\n测试：全量304通过/5跳过，CARLA SDK专项14通过。所有失败记录保留。历史资产完整性以`m5m_integrity.json`为准。\n\n图：[风险覆盖](m5m_figures/m5m_risk_coverage.png)、[接触空间](m5m_figures/m5m_contact_geometry.png)、[尾部失败样本](m5m_figures/m5m_contact_failures.png)、[残差Bias](m5m_figures/m5m_residual_bias.png)、[Shadow原始tick](m5m_figures/m5m_shadow_ticks.png)。\n"
    )


if __name__ == "__main__":
    main()
