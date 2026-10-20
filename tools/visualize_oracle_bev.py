"""Replay one committed frame and generate four-layer BEV + six-camera boxes."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from datasets.carla_dataset import CarlaDataset
from geometry.oracle_bev import depth_to_ego, height_bev
from geometry.visualization import bev_panel, overlay_boxes
from simulator.recorder import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, help='Recorded scene directory')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--bev-config', default='configs/bev.yaml')
    p.add_argument('--depth-convention', required=True, choices=['optical_z','ray_range'],
                   help='Explicit hypothesis; passing this flag is not empirical validation')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    ds = CarlaDataset(args.scene)
    sample, rgb, depths, objects = ds.read(args.index)
    cfg = yaml.safe_load(Path(args.bev_config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    points, panels = [], []
    per_camera = []
    for name, camera in ds.calibration['cameras'].items():
        pts = depth_to_ego(depths[name], camera['K'], camera['T_hs_ego_from_opt'],
                          depth_convention=args.depth_convention, stride=args.stride)
        points.append(pts)
        per_camera.append(height_bev(pts, cfg))
        panel = overlay_boxes(rgb[name], objects, camera['K'], camera['T_hs_ego_from_opt'])
        cv2.putText(panel, f'ORACLE {name}', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        panels.append(panel)
    counts = np.sum(per_camera,axis=0,dtype=np.uint32)
    np.savez_compressed(out/'oracle_bev.npz', observed_geometry=counts,
                        per_camera_counts=np.stack(per_camera))
    cv2.imwrite(str(out/'bev.png'), bev_panel(counts,cfg,objects))
    layer_panels = [bev_panel(counts[k:k+1],cfg,objects,scale=2) for k in range(len(counts))]
    cv2.imwrite(str(out/'height_layers.png'), np.concatenate(layer_panels,axis=1))
    while len(panels) % 3:
        panels.append(np.zeros_like(panels[0]))
    mosaic = np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in range(0,len(panels),3)],axis=0)
    cv2.imwrite(str(out/'cameras_boxes.png'),mosaic)
    atomic_json(out/'report.json', {'frame_id': sample['frame_id'], 'scene': str(ds.root),
        'mode': 'oracle_geometry', 'depth_convention_hypothesis': args.depth_convention,
        'depth_convention_verified': False, 'stride': args.stride,
        'calibration_id': sample['calibration_id'], 'bev_config': cfg,
        'valid_points': sum(len(p) for p in points), 'in_grid_points': int(counts.sum()),
        'representation': 'visible surface counts, not occupancy/free space'})
    print(out)


if __name__ == '__main__':
    main()
