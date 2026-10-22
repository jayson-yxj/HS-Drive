"""Class/occlusion support and uncertainty visualization, no tuning."""

import json
from pathlib import Path
import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tools.m5n_common import write, asset
from diagnostics.m5n_metrics import error


def main():
    rows = []
    empty = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for split in ["val", "sealed"]:
        ev = json.loads(Path(f"reports/m5n_eval_{split}.json").read_text())
        sup = json.loads(Path(f"runs/m5n_cache_{split}/supervision.json").read_text())
        cc = json.loads(Path(f"runs/m5n_cache_{split}/candidates.json").read_text())
        pr = json.loads(Path(f"runs/m5n_eval_{split}/predictions.json").read_text())
        rows.extend(
            dict(split=split, **v)
            for v in ev["strata"]
            if v["field"] in ["type", "road", "occlusion_proxy"]
            and v["method"] in ["plane", "direct", "geometry_prior"]
        )
        for seed, ax in zip([101, 102], axs):
            aa = [a for a in sup["actors"] if a["seed"] == seed and a["detected"]]
            x = [pr[a["candidate_id"]]["proxy_error_m"] for a in aa]
            y = [
                error(
                    pr[a["candidate_id"]]["positions"]["direct"], a["box_center_ego"]
                )["xy"]
                for a in aa
            ]
            ax.scatter(x, y, label=split, alpha=0.6, s=15)
            ax.set_xlabel("Empirical predicted XY error (m)")
            ax.set_ylabel("Actual simulator center XY error (m)")
            ax.set_title(f"seed{seed}")
            ax.legend()
        # Empty configurations identified from raw manifests, no assumption about physical false objects.
        from tools.m5n_common import rows as getrows

        rr = {r["sample_id"]: r for r in getrows(split)}
        empty_ids = []
        for sid, r in rr.items():
            man = json.loads(
                (Path(r["raw_frame"]).parent.parent / "manifest.json").read_text()
            )
            if not man["configurations"][r["configuration"]]:
                empty_ids.append(sid)
        for seed in [101, 102]:
            empty.append(
                dict(
                    split=split,
                    seed=seed,
                    empty_removed_frames=len(empty_ids),
                    candidate_count=sum(
                        c["seed"] == seed and c["sample_id"] in empty_ids for c in cc
                    ),
                    true_false_object_count=None,
                    reason="Empty scripted target configuration still contains background/parked real objects; no complete scene physical truth",
                )
            )
    fig.tight_layout()
    fig.savefig("reports/m5n_figures/m5n_uncertainty.png", dpi=160)
    plt.close(fig)
    write(
        "reports/m5n_stratum_support.json",
        dict(
            kind="offline_frozen_class_occlusion_proxy_slope_analysis",
            records=rows,
            empty_removed=empty,
            limits="All results conditional on reference association. Small classes/occlusion strata not scene-independent causal evidence. Slopes selected geometry-only, sealed mild slope~1.27deg not broad terrain validation.",
        ),
    )
    p = Path("reports/m5n_summary.md")
    text = p.read_text()
    text += "\n补充：新封存52帧/4位置、75个已知可见Actor-view；全部配对与Snapshot检查通过。盒中心和新显式local参数的最大中心差约0.000029m，验证的是仿真Box定义，不是Mesh真值。\n\nValidation误差预测与实际误差Spearman约0.412/0.101，封存约0.391/0.235，跨种子稳定性不足。按事后排序诊断能同时满足5m尾部和严重错误率的最大覆盖仅Val17.2%/22.6%、封存9.3%/13.3%，均不能作为Test新阈值。\n\n几何先验在封存也存在Max50.99/66.52m极端错误，不能据平均值较低升级它；本轮任何分支都没有获得规划许可。\n\n[不确定性与真实误差](m5n_figures/m5n_uncertainty.png)，类别、遮挡代理、坡度与Empty/Removed完整分母见`m5n_stratum_support.json`。\n"
    p.write_text(text)
    p = Path("docs/M5N_IMPLEMENTATION.md")
    p.write_text(
        p.read_text()
        + "\n补充可复现检查：`tools.m5n_inference_equivalence`验证完整RGB路径与缓存评测位置差小于2mm；`tools.m5n_sealed_reference_detail`验证显式Box局部参数；`tools.m5n_supplement`生成类别/Empty/置信度图。以上均不训练或选择模型。运行`m5n_finalize`前完成这些后处理。\n"
    )


if __name__ == "__main__":
    main()
