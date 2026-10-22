"""Actual measured scaling/error/support figures, without inferred missing results."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path("reports/m5o_figures")
    out.mkdir(exist_ok=True)
    frozen = json.loads(
        Path("reports/m5o_oracle_candidate_diagnostic.json").read_text()
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(frozen["records"]):
        vals = [
            r["candidate_stage_missed"],
            r["matched_direct_over5"],
            r["matched_direct_within5"],
        ]
        ax[i].bar(
            ["Missed candidate", "Matched >5m", "Matched <=5m"],
            vals,
            color=["#cb5555", "#d99638", "#4e9a84"],
        )
        ax[i].set_title(f"Historical regression seed {r['seed']}, n={r['visible']}")
        ax[i].set_ylabel("Actor views (correlated)")
    fig.tight_layout()
    if not (out / "m5o_regression_failure_decomposition.png").exists():
        fig.savefig(out / "m5o_regression_failure_decomposition.png", dpi=140)
    plt.close(fig)
    path = Path("reports/m5o_eval_val.json")
    if path.exists():
        d = json.loads(path.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for seed in [101, 102]:
            for method in ["direct", "geometry_prior"]:
                rr = sorted(
                    [
                        r
                        for r in d["records"]
                        if r["seed"] == seed and r["method"] == method
                    ],
                    key=lambda r: r["fraction"],
                )
                for j, key in enumerate(["mean", "p95"]):
                    ax[j].plot(
                        [r["fraction"] * 100 for r in rr],
                        [r["xy"][key] for r in rr],
                        marker="o",
                        label=f"{method} {seed}",
                    )
                    ax[j].set_ylabel(f"Validation XY {key} (m)")
                    ax[j].set_xlabel("Train location groups (%)")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "m5o_data_scaling.png", dpi=140)
        plt.close(fig)
    for split in ["calibration", "sealed"]:
        path = Path(f"reports/m5o_eval_{split}.json")
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for r in d["risks"]:
            x = [p["achieved_coverage"] for p in r["fixed_coverage"]]
            ax[0].plot(
                x,
                [p["xy"]["p95"] for p in r["fixed_coverage"]],
                marker="o",
                label=str(r["seed"]),
            )
            ax[1].plot(
                x,
                [p["xy"]["severe5m_rate"] for p in r["fixed_coverage"]],
                marker="o",
                label=str(r["seed"]),
            )
        for a in ax:
            a.set_xlabel("Coverage / all visible actor views")
            a.legend()
        ax[0].set_ylabel("XY P95 (m)")
        ax[1].set_ylabel("Error >5m fraction")
        fig.suptitle(f"{split}: descriptive risk-ordered curves, no test tuning")
        fig.tight_layout()
        fig.savefig(out / f"m5o_{split}_risk_coverage.png", dpi=140)
        plt.close(fig)
    p = Path("reports/m5o_training.json")
    if p.exists():
        d = json.loads(p.read_text())
        if d["status"] == "Completed":
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for r in d["records"]:
                if r["fraction"] != 1:
                    continue
                h = [h for h in r["history"] if "loss" in h]
                v = [h for h in r["history"] if "val_macro_xy" in h]
                label = f"{r['arm']} {r['seed']}"
                ax[0].plot([x["step"] for x in h], [x["loss"] for x in h], label=label)
                ax[1].plot(
                    [x["step"] for x in v], [x["val_macro_xy"] for x in v], label=label
                )
            ax[0].set_ylabel("Train loss")
            ax[1].set_ylabel("Validation location macro XY (m)")
            for a in ax:
                a.set_xlabel("Step")
                a.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "m5o_training.png", dpi=140)
            plt.close(fig)


if __name__ == "__main__":
    main()
