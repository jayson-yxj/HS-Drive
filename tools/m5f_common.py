"""Checked inference readers; GT resides only in explicit diagnostic loader."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from datasets.inputs import validate_tensor_inputs
from simulator.recorder import file_hash


def input_path(row):
    p=Path(row['input'])
    if p.is_absolute():return p
    roots=['m5d_index','m5e_index','m5f_test_index','m5c_index','m5b_index','m5a_index']
    for root in roots:
        p=Path('/root/gpufree-data/hs-drive-data/processed',root,row['input'])
        if p.exists() and file_hash(p)==row['input_sha256']:return p
    raise FileNotFoundError(row['input'])


def inputs(row,device='cuda'):
    path=input_path(row)
    if file_hash(path)!=row['input_sha256']:raise ValueError('inputSHA')
    r=json.loads(path.read_text())
    if set(r)!={'sample_id','images','intrinsics','cam_to_ego','ego_pose','timestamp_s'}:raise ValueError('GTnotallowed')
    images=[]
    for x in r['images']:
        p=Path(x['path'])
        if not p.name.endswith('_rgb.png') or file_hash(p)!=x['sha256']:raise ValueError('RGBonly')
        rgb=cv2.imread(str(p))[:,:,::-1].copy().transpose(2,0,1);images.append(rgb)
    d=dict(images=torch.from_numpy(np.stack(images)).float()/255,intrinsics=torch.tensor(r['intrinsics'],dtype=torch.float32),cam_to_ego=torch.tensor(r['cam_to_ego'],dtype=torch.float32),ego_pose=torch.tensor(r['ego_pose'],dtype=torch.float64),ego_state=torch.zeros(7),timestamps_s=torch.tensor(r['timestamp_s'],dtype=torch.float64),route_command=torch.tensor(0,dtype=torch.int64),valid_cameras=torch.ones(6,dtype=torch.bool));d={k:v.unsqueeze(0).to(device) for k,v in d.items()};validate_tensor_inputs(d);return d


def pixel_label(row):
    if file_hash(row['pixel_label'])!=row['pixel_sha256']:raise ValueError('pixelSHA')
    with np.load(row['pixel_label']) as z:return {k:z[k] for k in z.files}


def checkpoint(record):
    if file_hash(record['checkpoint'])!=record['sha256']:raise ValueError('checkpointSHA')
    return torch.load(record['checkpoint'],weights_only=True)
