"""Same official preprocessing, one CPU owner, separate unchanged GPU scatter."""
import threading
import time
import numpy as np
import torch
from datasets.m8j12_contract import CAMERAS,MODEL_KEYS,assert_observation,native_global_pose,measured_navigation_target


class OfficialPreprocessor:
    def __init__(self,cfg,camera):
        from mmdet.datasets.pipelines import Compose
        self.pipeline=Compose([x for x in cfg.test_pipeline if x['type']!='LoadMultiViewImageFromFiles'])
        self.camera=camera;self.thread=None
    def cpu(self,row,images):
        import cv2
        from mmcv.parallel import collate
        current=threading.get_ident()
        if self.thread is None:self.thread=current
        if self.thread!=current:raise RuntimeError('PreprocessorThreadOwnership')
        timing={};start=time.monotonic();assert_observation(row,images);rgb=[];encode=decode=0.
        for name in CAMERAS:
            t=time.monotonic();ok,jpeg=cv2.imencode('.jpg',images[name],[int(cv2.IMWRITE_JPEG_QUALITY),20]);encode+=time.monotonic()-t
            if not ok:raise RuntimeError('NativeJPEGEncodingFailed')
            t=time.monotonic();rgb.append(cv2.imdecode(jpeg,cv2.IMREAD_COLOR).astype(np.float32));decode+=time.monotonic()-t
        timing.update(jpeg_encode_ms=encode*1000,jpeg_decode_float_ms=decode*1000)
        t=time.monotonic();imu=np.asarray(row['imu']);status=np.zeros(10,np.float32)
        status[:3]=imu[:3]*[1,-1,1];status[3:6]=-imu[3:6];status[6]=row['speed_mps']
        data=dict(img=rgb,timestamp=float(row['timestamp_s']),lidar2img=[self.camera['projection'][x].copy() for x in CAMERAS],
            cam_intrinsic=[self.camera['K'][x].copy() for x in CAMERAS],lidar2global=native_global_pose(row['ego_world_matrix']),
            ego_status=status,gt_ego_fut_cmd=np.eye(6,dtype=np.float32)[row['navigation_command']-1],tp_near=measured_navigation_target(row),
            img_shape=[x.shape for x in rgb],ori_shape=[x.shape for x in rgb],
            aug_config=dict(resize=1.,resize_dims=(704,396),crop=(0,12,704,396),flip=False,rotate=0,rotate_3d=0))
        timing['metadata_ms']=(time.monotonic()-t)*1000
        for transform in self.pipeline.transforms:
            t=time.monotonic();data=transform(data);timing[type(transform).__name__+'_ms']=(time.monotonic()-t)*1000
        if set(data)!=MODEL_KEYS:raise RuntimeError('UnexpectedModelInputKeys')
        t=time.monotonic();batch=collate([data],samples_per_gpu=1);timing['collate_ms']=(time.monotonic()-t)*1000
        timing['cpu_total_ms']=(time.monotonic()-start)*1000
        return batch,timing
    @staticmethod
    def gpu(cpu):
        from mmcv.parallel import scatter
        t=time.monotonic();batch=scatter(cpu,[0])[0];torch.cuda.synchronize()
        assert batch['img'].shape==(1,6,3,384,704) and batch['img'].dtype==torch.float32
        return batch,(time.monotonic()-t)*1000


def equal_nested(a,b,path='batch'):
    """Exact values and layouts for every tensor/metadata leaf."""
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride() and a.device==b.device,path
        assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.strides==b.strides,path
        assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:equal_nested(a[k],b[k],path+'.'+k)
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_nested(x,y,path+'.'+str(i))
    else:assert type(a)==type(b) and a==b,(path,a,b)
