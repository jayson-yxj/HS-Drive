"""OpenCV-only diagnostic rendering; all artifacts are Oracle/Privileged."""
import cv2
import numpy as np
from .reference import transform_points, project_optical_points

# CARLA bounding-box corner order (get_world_vertices).
BOX_EDGES = ((0,1),(1,3),(3,2),(2,0),(0,4),(4,5),(5,1),(5,7),(7,6),(6,4),(6,2),(7,3))


def overlay_boxes(rgb, objects, K, cam_to_ego):
    out = rgb[..., ::-1].copy()
    for obj in objects:
        if obj.get('is_ego'):
            continue
        p = transform_points(obj['corners_hs_ego_m'], np.linalg.inv(cam_to_ego))
        for a, b in BOX_EDGES:
            pa, pb = p[a].copy(), p[b].copy()
            near = 0.1
            if max(pa[2], pb[2]) < near:
                continue
            if pa[2] < near:
                pa += (pb-pa) * ((near-pa[2])/(pb[2]-pa[2]))
            if pb[2] < near:
                pb += (pa-pb) * ((near-pb[2])/(pa[2]-pb[2]))
            uv = project_optical_points([pa,pb], K)
            uv = np.clip(uv, -100000, 100000).astype(int)
            ok, a2, b2 = cv2.clipLine((0,0,out.shape[1],out.shape[0]), tuple(uv[0]), tuple(uv[1]))
            if ok:
                cv2.line(out, a2, b2, (0,255,80), 1, cv2.LINE_AA)
    return out


def bev_panel(counts, config, objects=(), scale=4):
    value = np.log1p(counts.sum(axis=0).astype(float))
    value = (255*value/max(value.max(),1)).astype(np.uint8)
    out = cv2.applyColorMap(value[::-1, ::-1].copy(), cv2.COLORMAP_VIRIDIS)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    nx, ny = counts.shape[1:]
    def xy(points):
        p = np.asarray(points)
        # Display +X up, +Y left; array remains [Z,X,Y] on disk.
        cols = ny - (p[:,1]-config['y_range_m'][0])/config['resolution_m']
        rows = nx - (p[:,0]-config['x_range_m'][0])/config['resolution_m']
        return np.rint(np.stack([cols, rows],axis=1)*scale).astype(np.int32)
    for obj in objects:
        hull = cv2.convexHull(xy(obj['corners_hs_ego_m']))
        cv2.polylines(out, [hull], True, (255,255,255) if obj.get('is_ego') else (0,255,80), 1)
    origin, ahead = xy([[0,0],[8,0]])
    cv2.arrowedLine(out, tuple(origin), tuple(ahead), (0,0,255), 2)
    cv2.putText(out, '+X up / +Y left', (8,20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(out, 'ORACLE counts; empty != free', (8,40), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
    return out
