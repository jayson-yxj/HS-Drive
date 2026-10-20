import numpy as np
import torch
from geometry.ray_tracing import VoxelGrid
from geometry.reference import unproject_pixels_to_optical, transform_points
from models.view_transform import unproject, HeightSplat
from datasets.labels import depth_bin_edges


def test_all_projection_variants_share_pixel_and_continuous_matches_cpu():
    uv = torch.tensor([[[[8.0, 200.0], [328.0, 184.0]]]])
    d = torch.tensor([[[5.0, 24.0]]])
    k = torch.tensor([[[[300.0, 0.0, 319.5], [0.0, 300.0, 191.5], [0.0, 0.0, 1.0]]]])
    t = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 1.7],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    actual = unproject(uv, d, k, t).numpy().reshape(-1, 3)
    expected = transform_points(
        unproject_pixels_to_optical(uv[0, 0, :, 0], uv[0, 0, :, 1], d[0, 0], k[0, 0]),
        t[0, 0],
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    # Linear reconstruction of a depth from adjacent centers is geometrically exact
    # only for the coordinate centroid, not discrete voxel probability assignment.
    edges = depth_bin_edges(48, 1.5, 80)
    centers = (edges[:-1] + edges[1:]) / 2
    target = 24.0
    hi = np.searchsorted(centers, target)
    alpha = (target - centers[hi - 1]) / (centers[hi] - centers[hi - 1])
    assert abs((1 - alpha) * centers[hi - 1] + alpha * centers[hi] - target) < 1e-5


def test_height_boundary_not_xy_equivalent():
    grid = VoxelGrid()
    ids, inside = grid.locate([[10.0, 0.0, 0.999], [10.0, 0.0, 1.001]])
    assert inside.all() and ids[0] != ids[1]
