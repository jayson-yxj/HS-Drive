"""Synthetic 48LID/0.5m lattice phase sensitivity; not actual scene performance."""

from pathlib import Path
import numpy as np
from datasets.labels import depth_bin_edges
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json, file_hash


def main():
    out = Path("reports/m5c_quantization_lattice.json")
    if out.exists():
        raise FileExistsError(out)
    edges = depth_bin_edges(48, 1.5, 80)
    c = (edges[1:] + edges[:-1]) / 2
    g = VoxelGrid()
    records = []
    for lo, hi in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]:
        z = np.linspace(lo, hi, 10001, endpoint=False)
        nearest = c[np.abs(z[:, None] - c).argmin(1)]
        gt = np.stack([z + 1, np.full_like(z, 0.1), np.full_like(z, 1.2)], -1)
        q = gt.copy()
        q[:, 0] = nearest + 1
        r, _ = g.locate(gt)
        s, _ = g.locate(q)
        records.append(
            dict(
                optical_z_range=[lo, hi],
                n=len(z),
                correct_voxel_fraction=float((r == s).mean()),
                mean_xy_error_m=float(abs(z - nearest).mean()),
                max_xy_error_m=float(abs(z - nearest).max()),
                bin_width_min_m=float(
                    np.diff(edges)[np.clip(np.searchsorted(edges, z) - 1, 0, 47)].min()
                ),
                bin_width_max_m=float(
                    np.diff(edges)[np.clip(np.searchsorted(edges, z) - 1, 0, 47)].max()
                ),
            )
        )
    atomic_json(
        out,
        dict(
            kind="synthetic_fixed_forward_camera_uniform_depth_sampling",
            records=records,
            source_sha256=file_hash(__file__),
            limits="ConstantY/Z and fixed1m camera translation; uniform depth sweeps, not actual camera/actor distribution. Illustrates grid phase and binwidth only. Continuous interpolated centroid could recover GT if sub-bin fraction were known; fraction is not available from unchanged categorical model. No claim of implemented or learned improvement.",
            depth_centers=c.tolist(),
            depth_edges=edges.tolist(),
        ),
    )


if __name__ == "__main__":
    main()
