"""Measured postprocessing after sealed capture and isolated GPU budget."""

from tools.m5n_develop import run


def main():
    run(["tools.m5n_inference_equivalence"], "inference_equivalence.log")
    run(["tools.m5n_diagnostics"], "diagnostics.log")
    run(["tools.m5n_report"], "report.log")
    run(["tools.m5n_figures"], "figures_final.log")


if __name__ == "__main__":
    main()
