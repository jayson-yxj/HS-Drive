"""Layer-resolved raw predictions: expose unsupported high-layer positive scores."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_height_visuals.json")
    if dest.exists():
        raise FileExistsError(dest)
    report = json.loads(
        Path("reports/m5c_absolute_obstacle_detection.json").read_text()
    )
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5c_test_index")
    rows = json.loads((index / "index.json").read_text())["samples"]
    i = next(
        i
        for i, r in enumerate(rows)
        if r["group"] == "Town01_c3" and r["configuration"] == 3
    )
    refs = json.loads((index / "m5c_surface_index.json").read_text())["records"]
    ref = next(r for r in refs if r["sample_id"] == rows[i]["sample_id"])
    with np.load(ref["path"]) as z:
        valid = z["valid"]
        gt = z["target"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    names = ["C_pair", "D_direct"]
    edges = [-2, -0.5, 1, 2.5, 4]
    for layer in range(4):
        img = np.zeros((160, 120, 3))
        img[valid[layer]] = [0.3, 0.3, 0.3]
        img[valid[layer] & gt[layer]] = [1, 0.2, 0]
        axes[0, layer].imshow(img, origin="lower")
        axes[0, layer].set_title(f"GT surface z[{edges[layer]},{edges[layer+1]})")
    for row, name in enumerate(names, 1):
        r = next(r for r in report["records"] if r["name"] == name and r["seed"] == 71)
        with np.load(r["predictions"]) as z:
            score = z["score"][i]
        for layer in range(4):
            axes[row, layer].imshow(
                score[layer], origin="lower", cmap="magma", vmin=0, vmax=1
            )
            axes[row, layer].set_title(f"{name} layer{layer}, raw score")
    fig.suptitle(
        "Fixed Town01_c3 config3; black GT = unknown, not free; raw scores include unknown"
    )
    fig.tight_layout()
    path = Path("reports/m5c_figures/m5c_height_layers.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    atomic_json(
        dest,
        dict(
            kind="actual_test_prediction_visualization",
            figure=str(path),
            sha256=file_hash(path),
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
