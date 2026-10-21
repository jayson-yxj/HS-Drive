"""Opt-in six-camera RGB buffer packet; no simulator tick or control ownership."""
from dataclasses import dataclass
import numpy as np
import torch
from models.m5j_candidate import CAMERAS

@dataclass
class LatestRGBPacket:
    max_age_s: float=.25
    last_frame: int=-1
    last_timestamp: float=-1.

    def accept(self, frames, timestamps, now_s):
        if len(frames)!=6 or len(timestamps)!=6:raise ValueError('Six synchronized cameras required')
        if len(set(frames))!=1 or max(timestamps)-min(timestamps)>1e-6:raise ValueError('Unsynchronized packet')
        t=float(timestamps[0]);frame=int(frames[0])
        if not np.isfinite([t,now_s]).all() or t<=0 or now_s<t or now_s-t>self.max_age_s:raise ValueError('Future, invalid or stale packet')
        if frame<=self.last_frame or t<=self.last_timestamp:raise ValueError('Duplicate/out of order packet')
        self.last_frame=frame;self.last_timestamp=t
        return t

def bgra_to_pinned(buffers, *, height=384,width=640,camera_names=CAMERAS):
    if tuple(camera_names)!=CAMERAS or len(buffers)!=6:raise ValueError('Camera identity/order')
    out=torch.empty((1,6,3,height,width),dtype=torch.uint8,pin_memory=True)
    for ci,buffer in enumerate(buffers):
        raw=np.frombuffer(buffer,dtype=np.uint8)
        if raw.size!=height*width*4:raise ValueError('Incorrect BGRA byte count')
        # Copy into owned pinned storage before CARLA callback buffer lifetime ends.
        out[0,ci].copy_(torch.from_numpy(raw.reshape(height,width,4)[...,[2,1,0]].transpose(2,0,1).copy()))
    return out

def to_gpu(pinned_rgb, metadata):
    if not pinned_rgb.is_pinned() or pinned_rgb.dtype!=torch.uint8:raise ValueError('Owned pinned uint8 RGB required')
    if 'images'in metadata:raise ValueError('Metadata must not replace RGB')
    from datasets.inputs import validate_tensor_inputs
    result={k:v.to('cuda',non_blocking=True)for k,v in metadata.items()}
    result['images']=pinned_rgb.to('cuda',non_blocking=True).float().div_(255)
    validate_tensor_inputs(result)
    return result
