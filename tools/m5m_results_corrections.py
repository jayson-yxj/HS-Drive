"""Evidence-driven reporting precision; no experimental selection changes."""

import json
from pathlib import Path
from tools.m5l_common import asset


def main():
    p = Path("reports/m5m_vehicle_residual_gating.json")
    d = json.loads(p.read_text())
    d["attribution"] = asset("reports/m5m_residual_attribution.json")
    d["reason"] = (
        "NoGo: 35–80m all-surface and per-position protection fail, dominated by unassociated foreground (208 rays), NOT reliable background (575 rays). RGB gating protects reliable background substantially, but cannot identify safe Vehicle-only regions. Do not extend residual training."
    )
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    p = Path("reports/m5m_acceptance.json")
    d = json.loads(p.read_text())
    d["rationale"][
        "vehicle"
    ] = "Reliable background largely protected; all-far-surface degradation remains in unassociated foreground, including per-position failure. NoGo under unchanged gates."
    d["tests"] = dict(
        full="304 passed, 5 skipped",
        carla_sdk="14 passed",
        full_log=asset("runs/m5m_logs/full_tests_delivery.log"),
        sdk_log=asset("runs/m5m_logs/sdk_tests.log"),
    )
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    p = Path("reports/m5m_ground_plane_diagnostics.json")
    d = json.loads(p.read_text())
    d["sealed_diagnostics"] = asset("reports/m5m_sealed_ground.json")
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    p = Path("reports/m5m_shadow_runtime.json")
    d = json.loads(p.read_text())
    d["isolated_ipc_probe"] = asset("reports/m5m_ipc_probe.json")
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    p = Path("reports/m5m_summary.md")
    s = (
        p.read_text()
        .replace(
            "背景和逐场景门禁失败，停止这类残差训练。",
            "分层查明：35–80m的783条可靠射线中，575条背景仅由−9.491变为−9.518m；208条未关联Actor的前景由−10.778恶化为−20.813m。不能把这部分直接称背景或已确认Vehicle。全部远场及逐场景门禁仍失败，停止这类残差训练。",
        )
        .replace(
            "但远场背景保护失败。",
            "但全部远场表面与逐场景保护失败；可靠背景大体受到保护。",
        )
    )
    s += "\n补充：[轮胎/脚部候选投影](m5m_figures/m5m_anatomy_candidates.png)。物理轮/骨骼附近有渲染表面支持不等于精确接地，仍无Gold。\n"
    p.write_text(s)
    p = Path("docs/M5M_IMPLEMENTATION.md")
    s = (
        p.read_text()
        + "\n后处理复现：`tools.m5m_historical_audit`、`tools.m5m_residual_attribution`、`tools.m5m_ipc_probe`、`tools.m5m_anatomy_figures`、`tools.m5m_run_finish`、`tools.m5m_results_corrections`、`tools.m5m_integrity`。格式化后旧实验报告源SHA不同于最终文件时，pretest前哈希代表当时源码；格式化不改变数值方法。\n"
    )
    p.write_text(s)


if __name__ == "__main__":
    main()
