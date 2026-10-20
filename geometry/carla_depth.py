"""Raw depth must never be replaced by grayscale visualization data."""
import numpy as np
from .reference import decode_carla_depth_bgra


def decode_rgb_bgra(raw_data, height, width):
    return np.frombuffer(raw_data, np.uint8).reshape(height, width, 4)[..., 2::-1].copy()


def valid_depth(depth, min_m=1.5, max_m=80.0):
    return np.isfinite(depth) & (depth >= min_m) & (depth < max_m)
