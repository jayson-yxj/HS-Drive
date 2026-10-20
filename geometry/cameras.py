"""Pinhole camera geometry with integer pixel centers."""
from .reference import intrinsics_from_hfov, project_optical_points, unproject_pixels_to_optical


def resize_rgb_and_intrinsics(rgb, K, output_wh):
    """Resize RGB with OpenCV half-pixel mapping and update K consistently.

    Do not apply area averaging to depth labels: it blends depth discontinuities.
    """
    import cv2
    import numpy as np
    width,height=output_wh
    if width<1 or height<1:raise ValueError('Output dimensions must be positive')
    sx,sy=width/rgb.shape[1],height/rgb.shape[0]
    A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])
    method=cv2.INTER_AREA if sx<=1 and sy<=1 else cv2.INTER_LINEAR
    return cv2.resize(rgb,(width,height),interpolation=method),A@np.asarray(K)
