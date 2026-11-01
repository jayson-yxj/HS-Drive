"""Candidate-only native payload copies and finite rendered-marker measurement."""
from dataclasses import dataclass
from queue import Queue, Full, Empty
from threading import Lock
import time
import numpy as np
from simulator.m8j13_sensor_audit import RawImageStamp


@dataclass(frozen=True)
class CopiedImage:
    stamp: RawImageStamp
    width: int
    height: int
    bgra: bytes
    copied_monotonic_s: float


class CandidateInbox:
    def __init__(self, identity, capacity=96):
        self.identity=identity; self.queue=Queue(capacity); self.lock=Lock()
        self.active=True; self.overflow=0; self.after_close=0

    def callback(self, name, actor_id):
        identity=self.identity
        def receive(image):
            arrival=time.monotonic()
            stamp=RawImageStamp(*identity,name,actor_id,int(image.frame),float(image.timestamp),
                tuple(tuple(float(v) for v in row) for row in image.transform.get_matrix()),arrival)
            # Independent bytes at callback entry; no disk, conversion or hash.
            payload=CopiedImage(stamp,int(image.width),int(image.height),bytes(image.raw_data),time.monotonic())
            with self.lock:
                if not self.active:self.after_close+=1;return
                try:self.queue.put_nowait(payload)
                except Full:self.overflow+=1
        return receive

    def drain(self):
        while True:
            try:yield self.queue.get_nowait()
            except Empty:return

    def close(self):
        with self.lock:self.active=False


def marker_pixels(bgra, width, height, camera_matrix, center_world, opposite_world, fov, cfg):
    """Fixed red-body test against snapshot positions, never rendered overlays."""
    image=np.frombuffer(bgra,np.uint8).reshape(height,width,4)
    inv=np.linalg.inv(np.asarray(camera_matrix)); focal=width/(2*np.tan(np.radians(fov)/2))
    def score(point):
        x,y,z,_=inv@np.r_[point,1.]
        if x<=0:return dict(count=0,center=None)
        u=width/2+focal*y/x;v=height/2-focal*z/x
        a,b=cfg['roi_halfwidth_px'],cfg['roi_halfheight_px']
        patch=image[max(0,int(v-b)):min(height,int(v+b)),max(0,int(u-a)):min(width,int(u+a))]
        blue,green,red=[patch[:,:,i].astype(float) for i in range(3)]
        mask=(red>cfg['red_min'])&(red>cfg['red_channel_ratio']*green)&(red>cfg['red_channel_ratio']*blue)
        return dict(count=int(mask.sum()),center=[float(u),float(v)])
    expected=score(center_world);opposite=score(opposite_world)
    return dict(expected=expected,opposite=opposite,
        matched=expected['count']>=cfg['min_red_pixels'] and opposite['count']<=cfg['opposite_max_ratio']*expected['count'])


def deadline_decision(input_sim, input_wall, now_sim, now_wall, max_sim=.3, max_wall=.3):
    from simulator.m8j13_safety import PlanStamp, freshness, SafetyGate
    stamp=PlanStamp('sensor_diagnostic',0,input_sim,input_wall,input_wall,now_wall,'no_model')
    reason=freshness(stamp,'sensor_diagnostic',0,now_sim,now_wall,max_sim,max_wall)
    # Numeric independent stop command only; never dispatched to a vehicle.
    stop=SafetyGate({}).stop(reason,now_sim,now_wall) if reason else None
    return dict(accepted=reason is None,reason=reason,stop_command=stop,actuated=False,
        sim_age_s=now_sim-input_sim,wall_age_s=now_wall-input_wall)
