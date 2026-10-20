"""Independent M4.6 small-sample loader; inference never reads privileged files."""

import json
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from datasets.inputs import LearnedInputs
from datasets.training import DataItem
from datasets.occupancy_labels import validate_occupancy
from simulator.recorder import file_hash


def checked(path, sha):
    path = Path(path)
    if file_hash(path) != sha:
        raise ValueError("Occupancy input/label checksum mismatch")
    return path


class OccupancyDataset(Dataset):
    def __init__(self, index, split, *, supervision=True):
        self.root = Path(index)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["schema_version"] != "hsd-m46-training-index-v1":
            raise ValueError("Unsupported occupancy index")
        self.index = json.loads(
            checked(self.root / "index.json", manifest["index_sha256"]).read_text()
        )
        self.rows = [r for r in self.index["samples"] if r["split"] == split]
        self.supervision = supervision
        groups = {}
        for r in self.index["samples"]:
            if r["group"] in groups and groups[r["group"]] != r["split"]:
                raise ValueError("Scene group crosses split")
            groups[r["group"]] = r["split"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        r = json.loads(
            checked(self.root / row["input"], row["input_sha256"]).read_text()
        )
        if set(r) != {
            "sample_id",
            "images",
            "intrinsics",
            "cam_to_ego",
            "ego_pose",
            "timestamp_s",
        }:
            raise ValueError("Input schema forbids GT")
        images = []
        for ref in r["images"]:
            path = checked(ref["path"], ref["sha256"])
            if path.suffix != ".png" or not path.name.endswith("_rgb.png"):
                raise ValueError("RGB references required")
            rgb = cv2.imread(str(path))
            if rgb is None or rgb.shape != (384, 640, 3):
                raise ValueError("RGB shape")
            images.append(rgb[:, :, ::-1].copy().transpose(2, 0, 1))
        inputs = LearnedInputs(
            dict(
                images=torch.from_numpy(np.stack(images)).float() / 255,
                intrinsics=torch.tensor(r["intrinsics"], dtype=torch.float32),
                cam_to_ego=torch.tensor(r["cam_to_ego"], dtype=torch.float32),
                ego_pose=torch.tensor(r["ego_pose"], dtype=torch.float64),
                ego_state=torch.zeros(7),
                timestamps_s=torch.tensor(r["timestamp_s"], dtype=torch.float64),
                route_command=torch.tensor(0, dtype=torch.int64),
                valid_cameras=torch.ones(6, dtype=torch.bool),
            )
        )
        targets = None
        if self.supervision:
            with np.load(
                checked(row["label"], row["label_sha256"]), allow_pickle=False
            ) as arr:
                values = {k: arr[k] for k in arr.files}
            validate_occupancy(values)
            if values["occupancy_state"].shape != (4, 160, 120):
                raise ValueError("BEV shape")
            targets = {
                k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.uint32 else v)
                for k, v in values.items()
            }
        return DataItem(
            inputs,
            targets,
            dict(sample_id=row["sample_id"], group=row["group"], split=row["split"]),
        )
