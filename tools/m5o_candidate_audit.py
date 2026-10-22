"""All-candidate burden and per-scene montage; semantic proxies are not physical FP."""

import argparse, json
from collections import Counter
from pathlib import Path
import cv2, numpy as np
from tools.m5o_common import rows, write, asset, provenance
from datasets.m5o_benchmark import quality


def category(label):
    if label["matched"]:
        return "matched"
    hist = {int(k): v for k, v in label["semantic_histogram"].items()}
    n = max(sum(hist.values()), 1)
    if sum(hist.get(k, 0) for k in [12, 13, 14, 15, 16, 17, 18, 19]) / n >= 0.5:
        return "unrecorded_or_unassociated_target_surface"
    if sum(hist.get(k, 0) for k in [20, 21]) / n >= 0.5:
        return "unknown_static_dynamic_surface"
    if (
        sum(hist.get(k, 0) for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 25, 26, 27, 28])
        / n
        >= 0.9
    ):
        return "rendered_background_semantic_proxy"
    return "mixed_or_unexplained"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    root = Path(f"runs/m5o_cache_{a.split}")
    cc = json.loads((root / "candidates.json").read_text())
    sup = json.loads((root / "supervision.json").read_text())
    labels = {l["candidate_id"]: l for l in sup["labels"]}
    records = []
    figs = []
    figroot = Path("reports/m5o_figures")
    figroot.mkdir(exist_ok=True)
    for group in sorted({c["group"] for c in cc}):
        cs = [c for c in cc if c["group"] == group]
        s = dict(
            labels=[labels[c["candidate_id"]] for c in cs],
            actors=[x for x in sup["actors"] if x["group"] == group],
        )
        # quality requires original feature_index addressing; keep global candidate list, filter stats directly.
        for seed in [101, 102]:
            selected = [c for c in cs if c["seed"] == seed]
            aa = [x for x in s["actors"] if x["seed"] == seed]
            records.append(
                dict(
                    group=group,
                    seed=seed,
                    candidates=len(selected),
                    visible_actor_views=len(aa),
                    detected=sum(x["detected"] for x in aa),
                    split_actor_views=sum(x["split_components"] > 1 for x in aa),
                    categories=dict(
                        Counter(category(labels[c["candidate_id"]]) for c in selected)
                    ),
                )
            )
        rr = [r for r in rows(a.split) if r["group"] == group]
        images = []
        # Every frame six views, indexed contact sheets per group; no cherry-picking.
        for row in rr:
            f = json.loads(Path(row["raw_frame"]).read_text())
            raw = Path(row["raw_frame"]).parent.parent
            tiles = []
            for cam in row["camera_names"]:
                im = cv2.imread(str(raw / f["cameras"][cam]["rgb"]))
                for c in cs:
                    if (
                        c["sample_id"] != row["sample_id"]
                        or c["seed"] != 101
                        or c["camera"] != cam
                    ):
                        continue
                    lab = labels[c["candidate_id"]]
                    x, y, w, h = c["region"]
                    col = (0, 220, 0) if lab["matched"] else (0, 140, 255)
                    cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
                cv2.putText(
                    im,
                    f"cfg{row['configuration']} {cam}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                tiles.append(cv2.resize(im, (320, 192)))
            images.append(np.concatenate(tiles, axis=1))
        dst = figroot / f"m5o_{a.split}_{group}_candidates.jpg"
        assert not dst.exists()
        cv2.imwrite(str(dst), np.concatenate(images, axis=0))
        figs.append(asset(dst))
    out = dict(
        kind="offline_actual_CARLA_RGB_candidate_audit",
        split=a.split,
        aggregate=[quality(cc, sup, seed) for seed in [101, 102]],
        groups=records,
        categories=dict(Counter(category(l) for l in sup["labels"])),
        figures=figs,
        provenance=provenance(),
        limits="Unmatched != physical FP. Semantic background proxy may include poles/walls that are real physical objects outside target taxonomy. Recorded-target empty does not mean empty physical scene; renderer association proxy, not mesh truth.",
    )
    write(f"reports/m5o_candidate_quality_{a.split}.json", out)
    print(out["aggregate"], flush=True)


if __name__ == "__main__":
    main()
