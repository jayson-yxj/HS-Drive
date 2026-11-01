"""One opt-in candidate: two CPU workers, original per-camera operations.

Only JPEG, ResizeCropFlipImage, NormalizeMultiviewImage run before full input.
Each worker owns its transforms; final adaptor/Collect/collate stay serial.
"""
from concurrent.futures import ThreadPoolExecutor
import threading,time
import numpy as np
from datasets.m8j12_contract import CAMERAS,MODEL_KEYS,assert_observation,native_global_pose,measured_navigation_target
from tools.m8j13_preprocess import OfficialPreprocessor

class CameraPreprocessor:
    def __init__(self,cfg,camera):
        from mmdet.datasets.pipelines import Compose
        definitions=[x for x in cfg.test_pipeline if x['type']!='LoadMultiViewImageFromFiles']
        if [x['type'] for x in definitions]!=['ResizeCropFlipImage','NormalizeMultiviewImage','NuScenesSparse4DAdaptor','Collect']:raise ValueError('UnsupportedOfficialPipeline')
        self.early=definitions[:2];self.tail=Compose(definitions[2:]);self.camera=camera;self.local=threading.local()
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='camera-preprocess');self.active={};self.closed=False
    def begin(self,token):
        key=(token['slot'],token['generation'])
        if self.closed or key in self.active:raise RuntimeError('ClosedOrDuplicatePreparation')
        if len(self.active)>=2:raise RuntimeError('PartialObservationCapacity')
        self.active[key]={}
    def submit(self,slots,token,name):
        key=(token['slot'],token['generation']);jobs=self.active[key]
        if self.closed or name in jobs or name not in CAMERAS:raise RuntimeError('InvalidCameraPreparation')
        jobs[name]=self.pool.submit(self._one,slots,token,name)
    def _one(self,slots,token,name):
        import cv2
        from mmdet.datasets.pipelines import Compose
        if not hasattr(self.local,'pipeline'):self.local.pipeline=Compose(self.early)
        start=time.monotonic();cpu=time.thread_time();image,stamp=slots.read_camera(token,name);read=time.monotonic()
        ok,jpeg=cv2.imencode('.jpg',image,[int(cv2.IMWRITE_JPEG_QUALITY),20]);encoded=time.monotonic()
        if not ok:raise RuntimeError('JPEGEncodingFailed')
        rgb=cv2.imdecode(jpeg,cv2.IMREAD_COLOR).astype(np.float32);decoded=time.monotonic()
        data=dict(img=[rgb],lidar2img=[self.camera['projection'][name].copy()],cam_intrinsic=[self.camera['K'][name].copy()],
            img_shape=[rgb.shape],ori_shape=[rgb.shape],aug_config=dict(resize=1.,resize_dims=(704,396),crop=(0,12,704,396),flip=False,rotate=0,rotate_3d=0))
        stages={}
        for tr in self.local.pipeline.transforms:
            t=time.monotonic();data=tr(data);stages[type(tr).__name__+'_ms']=(time.monotonic()-t)*1000
        finished=time.monotonic()
        return dict(data=data,original=image,stamp=stamp,timing=dict(start=start,finish=finished,cpu_ms=(time.thread_time()-cpu)*1000,
            read_ms=(read-start)*1000,jpeg_encode_ms=(encoded-read)*1000,jpeg_decode_float_ms=(decoded-encoded)*1000,**stages))
    def assemble(self,slots,token):
        from mmcv.parallel import collate
        row=slots.committed_row(token);jobs=self.active[(token['slot'],token['generation'])]
        if set(jobs)!=set(CAMERAS):raise RuntimeError('IncompletePreparation')
        prepared={n:jobs[n].result() for n in CAMERAS};start=time.monotonic();cpu=time.thread_time()
        images={n:prepared[n]['original'] for n in CAMERAS};assert_observation(row,images)
        imu=np.asarray(row['imu']);status=np.zeros(10,np.float32);status[:3]=imu[:3]*[1,-1,1];status[3:6]=-imu[3:6];status[6]=row['speed_mps']
        data=dict(img=[prepared[n]['data']['img'][0] for n in CAMERAS],timestamp=float(row['timestamp_s']),
            lidar2img=[prepared[n]['data']['lidar2img'][0] for n in CAMERAS],cam_intrinsic=[prepared[n]['data']['cam_intrinsic'][0] for n in CAMERAS],
            img_shape=[prepared[n]['data']['img_shape'][0] for n in CAMERAS],ori_shape=[(396,704,3)]*6,
            img_norm_cfg=prepared[CAMERAS[0]]['data']['img_norm_cfg'],lidar2global=native_global_pose(row['ego_world_matrix']),
            ego_status=status,gt_ego_fut_cmd=np.eye(6,dtype=np.float32)[row['navigation_command']-1],tp_near=measured_navigation_target(row))
        stages={}
        for tr in self.tail.transforms:
            t=time.monotonic();data=tr(data);stages[type(tr).__name__+'_ms']=(time.monotonic()-t)*1000
        if set(data)!=MODEL_KEYS:raise RuntimeError('UnexpectedModelKeys')
        t=time.monotonic();batch=collate([data],samples_per_gpu=1);stages['collate_ms']=(time.monotonic()-t)*1000
        stages.update(serial_tail_start_wall_s=start,serial_tail_end_wall_s=time.monotonic(),serial_tail_ms=(time.monotonic()-start)*1000,serial_tail_cpu_ms=(time.thread_time()-cpu)*1000)
        return batch,dict(cameras={n:prepared[n]['timing'] for n in CAMERAS},tail=stages)
    def finish(self,token):
        jobs=self.active.pop((token['slot'],token['generation']))
        for job in jobs.values():
            try:job.result()
            except Exception:pass  # caller records main error; release only after readers stop
    def close(self):
        self.closed=True;self.pool.shutdown(wait=True);self.active.clear()
    gpu=staticmethod(OfficialPreprocessor.gpu)
