"""Supervision-only depth sampling. Never average across depth boundaries."""
import numpy as np


def depth_bin_edges(count=48, minimum=1.5, maximum=80.0, strategy='lid'):
    if count < 1 or not 0 < minimum < maximum:
        raise ValueError('Invalid depth bins')
    i = np.arange(count+1, dtype=np.float64)
    if strategy == 'lid':
        return minimum + (maximum-minimum)*i*(i+1)/(count*(count+1))
    if strategy == 'uniform':
        return np.linspace(minimum, maximum, count+1)
    raise ValueError('Unknown depth strategy')


def sample_depth(depth, *, output_wh=(640,384), stride=16, edges=None, max_patch_spread_m=.5):
    """Nearest native pixel to each feature-cell center, with its exact resized UV.

    Feature-cell center convention is explicit and must also be used by M4's
    view transform. depth_uv identifies the sampled ray, including rounding.
    The 3x3 native patch guard excludes uncertain occlusion boundaries.
    """
    width, height = output_wh
    if stride < 1 or width % stride or height % stride or depth.ndim != 2:
        raise ValueError('Depth sampling needs 2D input and stride-divisible output')
    if max_patch_spread_m < 0 or not np.isfinite(max_patch_spread_m):
        raise ValueError('Invalid depth edge threshold')
    edges = depth_bin_edges() if edges is None else np.asarray(edges)
    if len(edges)<2 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
        raise ValueError('Depth edges must strictly increase')
    nh, nw = depth.shape
    u = (np.arange(width//stride)+.5)*stride-.5
    v = (np.arange(height//stride)+.5)*stride-.5
    x = np.clip(np.floor((u+.5)*nw/width).astype(int),0,nw-1)
    y = np.clip(np.floor((v+.5)*nh/height).astype(int),0,nh-1)
    xx, yy = np.meshgrid(x,y)
    values = depth[yy,xx].astype(np.float32)
    patch = np.stack([depth[np.clip(yy+dy,0,nh-1),np.clip(xx+dx,0,nw-1)]
                      for dy in (-1,0,1) for dx in (-1,0,1)])
    finite_patch = np.isfinite(patch).all(axis=0)
    clean_patch = np.where(np.isfinite(patch),patch,0)
    valid = (np.isfinite(values) & (values>=edges[0]) & (values<edges[-1]) &
             finite_patch & ((clean_patch.max(axis=0)-clean_patch.min(axis=0))<=max_patch_spread_m))
    bins = np.searchsorted(edges,values,side='right').astype(np.int64)-1
    bins[~valid] = -100
    uv = np.stack(((xx+.5)*width/nw-.5,(yy+.5)*height/nh-.5),axis=-1).astype(np.float32)
    return {'depth_m':np.where(valid,values,0).astype(np.float32),'depth_valid':valid,
            'depth_bin':bins,'depth_uv':uv}
