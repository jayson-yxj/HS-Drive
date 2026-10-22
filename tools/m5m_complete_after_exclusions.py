"""Final finite inference and reporting after immutable paired-scene exclusions."""

from tools.m5m_sequence import run


def main():
    run(["tools.m5m_new_eval", "--phase", "sealed"], "sealed_evaluation.log")
    run(["tools.m5m_anatomy_audit", "--phase", "sealed"], "anatomy_sealed.log")
    run(["tools.m5m_sealed_ground"], "sealed_ground.log")
    run(["tools.m5m_contact_visibility_complete"], "visibility_complete.log")
    run(["tools.m5m_delivery_reports"], "delivery_reports.log")
    run(["tools.m5m_figures"], "figures_final.log")
    run(["tools.m5m_assemble"], "assemble.log")
    run(["tools.m5m_results_corrections"], "results_corrections.log")
    run(["tools.m5m_sync_source_notes"], "final_source_audit.log")


if __name__ == "__main__":
    main()
