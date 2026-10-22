"""Run after sealed pipeline completion; no concurrent GPU load during shadow."""

from pathlib import Path
from tools.m5m_sequence import run


def main():
    assert Path("reports/m5m_anatomy_sealed.json").exists()
    run(
        ["tools.m5m_capture_integrity", "--phase", "sealed"],
        "capture_integrity_sealed.log",
    )
    run(["tools.m5m_sealed_ground"], "sealed_ground.log")
    run(["tools.m5m_online_shadow"], "shadow.log")
    run(["tools.m5m_runtime_diagnostics"], "runtime_diagnostics.log")
    run(["tools.m5m_delivery_reports"], "delivery_reports.log")
    run(["tools.m5m_figures"], "figures_final.log")
    run(["tools.m5m_assemble"], "assemble.log")


if __name__ == "__main__":
    main()
