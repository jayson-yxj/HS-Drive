"""Final honest summary patch and schema/asset checks; retain measured decisions."""

import json
from pathlib import Path
from tools.m5l_common import asset


def main():
    p = Path("reports/m5m_summary.md")
    s = p.read_text()
    s = s.replace("另有逐配置自车移除参考。", "另有逐配置自车移除参考。")
    s = s.replace(
        "每位置9配置。原生实例轮廓",
        "每位置9配置；封存2帧配对姿态超差被剔除，主评测34帧，失败原数据保留。原生实例轮廓",
    )
    s += "\nShadow两轮平均16.4Hz，全部测量tick间隔均超过50ms；未触发严重退化停止不代表实时门禁通过。冷启动完整年龄约0.99/0.98秒，全部被拒绝；稳态端到端P95约133ms。\n"
    p.write_text(s)
    p = Path("reports/m5m_contact_visibility.json")
    d = json.loads(p.read_text())
    d["complete_native_visibility"] = asset("reports/m5m_visibility_complete.json")
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    p = Path("reports/m5m_acceptance.json")
    d = json.loads(p.read_text())
    d["sealed_data_quality"] = "Partial_2of36_pair_mismatch_excluded"
    d["sealed_eligible_frames"] = 34
    d["sealed_positions"] = 4
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    required = [
        "summary.md",
        "acceptance.json",
        "contact_reference_quality.json",
        "contact_visibility.json",
        "selective_localization.json",
        "risk_coverage.json",
        "vehicle_residual_gating.json",
        "ground_plane_diagnostics.json",
        "shadow_runtime.json",
    ]
    for suffix in required:
        assert Path("reports/m5m_" + suffix).exists(), suffix
    for p in Path("reports").glob("m5m*.json"):
        text = p.read_text()
        assert "NaN" not in text and "Infinity" not in text, p
        json.loads(text)
    for path in ["docs/M5M_IMPLEMENTATION.md", "docs/M5M_ARCHITECTURE_DECISION.md"]:
        assert Path(path).exists()
    for s in ["dev", "sealed"]:
        data = json.loads(Path(f"runs/m5m_contact_{s}.json").read_text())
        assert all(r["status"] == "Unknown" for r in data["outputs"])
        assert all(
            r["class_hint"] == "foreground_unknown_class" for r in data["outputs"]
        )
    print("Required reports/schema/Unknown-output checks passed")


if __name__ == "__main__":
    main()
