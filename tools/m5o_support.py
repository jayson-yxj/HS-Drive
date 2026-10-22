"""Audit real scene/target support before any new regression training."""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m5o_common import DATA, write, asset, provenance
from datasets.m5o_benchmark import support, quality, validate_split_isolation


def main():
    protocol = json.loads(Path("configs/m5o_protocol.json").read_text())
    config = json.loads(Path("configs/m5o_capture_dev_v3.json").read_text())
    groups = config["groups"]
    conflicts = validate_split_isolation(groups)
    records = []
    capture = []
    for split in ["train", "val", "calibration"]:
        pix = DATA / "processed" / f"m5o_{split}_pixels/index.json"
        root = Path(f"runs/m5o_cache_{split}")
        if not (root / "supervision.json").exists():
            records.append(dict(split=split, status="NotAvailable", passed=False))
            continue
        cc = json.loads((root / "candidates.json").read_text())
        sup = json.loads((root / "supervision.json").read_text())
        for seed in [101, 102]:
            s = support(cc, sup["labels"], seed)
            records.append(
                dict(
                    split=split,
                    seed=seed,
                    **s,
                    passed=s["unique_candidates"]
                    >= protocol["support"]["unique_candidates"][split]
                    and s["groups"] >= protocol["support"]["groups"][split],
                    class_claims={c: n >= 20 for c, n in s["by_class"].items()},
                    camera_claims={c: n >= 20 for c, n in s["by_camera"].items()},
                    quality=quality(cc, sup, seed),
                )
            )
        rr = json.loads(pix.read_text())["records"]
        good = bad = 0
        types = Counter()
        weather = Counter()
        geometry = Counter()
        ambiguous = []
        for r in rr:
            f = json.loads(Path(r["raw_frame"]).read_text())
            weather[r["weather"]] += 1
            geometry[r["geometry"]] += 1
            for actor in f["actors"]:
                if (
                    actor["is_ego"]
                    or np.linalg.norm(np.mean(actor["corners_hs_ego_m"], axis=0)[:2])
                    > 80
                ):
                    continue
                valid = actor["contact_reference"]["road_probe_found"]
                good += valid
                bad += not valid
                types[actor["type_id"]] += 1
            for cam in r["actors"]:
                ids = [
                    v["render_id"]
                    for actor in cam["actors"]
                    for v in actor["identity_candidates"]
                    if v["accepted"]
                ]
                if len(ids) != len(set(ids)):
                    ambiguous.append([r["sample_id"], cam["camera"]])
        capture.append(
            dict(
                split=split,
                frames=len(rr),
                groups=len({r["group"] for r in rr}),
                road_probe_supported=good,
                road_probe_missing=bad,
                actor_instances_by_blueprint=dict(types),
                weather=dict(weather),
                geometry=dict(geometry),
                ambiguous_render_identity_views=ambiguous,
                index=asset(pix),
            )
        )
    passed = not conflicts and all(r["passed"] for r in records)
    write(
        "reports/m5o_dataset_support.json",
        dict(
            kind="actual_CARLA_capture_and_offline_RGB_support",
            passed=passed,
            records=records,
            capture=capture,
            isolation_failures=conflicts,
            provenance=provenance(),
            protocol=asset("configs/m5o_protocol.json"),
            limits="Unique target views collapse identical group/camera/type/center (centimeter rounding), views of same actor are correlated. Randomized road layouts share pattern family; independent unit is road/location, not pixel/view. 300 candidates is planned support, not qualification. Some six directions cannot be filled on straight roads; class/camera under20 no independent claim. Road collision probe is placement support, not complete mesh/contact Gold. Sealed count assessed only after lock.",
        ),
    )
    print(
        "support",
        passed,
        [
            (r.get("split"), r.get("seed"), r.get("unique_candidates"), r.get("groups"))
            for r in records
        ],
    )


if __name__ == "__main__":
    main()
