"""Opt-in frozen official SparseDriveV2 construction; never a live driving Agent."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def protocol():
    return json.loads(Path('configs/m8j12_protocol.json').read_text())


def state_digest(model):
    result = {}
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(a.tobytes()).hexdigest()
    return result


def build():
    from mmcv import Config
    from mmdet.models import build_detector
    import projects.mmdet3d_plugin  # official registrations and extensions
    c = protocol()
    rt = Path(c['runtime_root'])
    cfg = Config.fromfile(str(rt / 'runtime-source' / c['config']))
    mappings = {}

    def paths(obj):
        if isinstance(obj, dict):
            return {k: paths(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(paths(x) for x in obj)
        if isinstance(obj, str) and obj.startswith('data/kmeans/'):
            dest = rt / 'artifacts' / Path(obj).name
            if not dest.is_file():
                raise FileNotFoundError(dest)
            mappings[obj] = str(dest)
            return str(dest)
        return obj

    model_cfg = paths(copy.deepcopy(cfg.model))
    model = build_detector(model_cfg)
    checkpoint = torch.load(rt / 'artifacts' / c['checkpoint'], map_location='cpu')
    # No init_weights()/pretrained download: every persistent inference tensor
    # must come from this single complete official checkpoint.
    result = model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.float().eval().requires_grad_(False)
    for module in model.modules():
        if getattr(module, 'fp16_enabled', False):
            raise RuntimeError('UnexpectedHalfPrecisionHook')
    return model, cfg, dict(path_mapping=mappings,missing=list(result.missing_keys),
                           unexpected=list(result.unexpected_keys))


def reset(model):
    model.head.det_head.instance_bank.reset()
    model.head.map_head.instance_bank.reset()
    model.head.motion_plan_head.instance_queue.reset()


def observation_batch(cfg, row, images, camera_contract):
    import cv2
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter
    from datasets.m8j12_contract import (CAMERAS, MODEL_KEYS, assert_observation,
                                        native_global_pose, measured_navigation_target)
    assert_observation(row, images)
    rgb = []
    for camera in CAMERAS:
        ok, jpeg = cv2.imencode('.jpg', images[camera], [int(cv2.IMWRITE_JPEG_QUALITY),20])
        if not ok:
            raise RuntimeError('NativeJPEGEncodingFailed')
        rgb.append(cv2.imdecode(jpeg, cv2.IMREAD_COLOR).astype(np.float32))
    imu = np.asarray(row['imu'])
    status = np.zeros(10, np.float32)
    status[:3] = imu[:3] * [1,-1,1]
    status[3:6] = -imu[3:6]
    status[6] = row['speed_mps']
    command = np.eye(6, dtype=np.float32)[row['navigation_command']-1]
    data = dict(img=rgb, timestamp=float(row['timestamp_s']),
        lidar2img=[camera_contract['projection'][x].copy() for x in CAMERAS],
        cam_intrinsic=[camera_contract['K'][x].copy() for x in CAMERAS],
        lidar2global=native_global_pose(row['ego_world_matrix']),
        ego_status=status, gt_ego_fut_cmd=command,
        tp_near=measured_navigation_target(row),
        img_shape=[x.shape for x in rgb], ori_shape=[x.shape for x in rgb],
        aug_config=dict(resize=1., resize_dims=(704,396),crop=(0,12,704,396),
                        flip=False,rotate=0,rotate_3d=0))
    # The official live Agent also omits the disk loader; images are already read.
    pipeline = Compose([x for x in cfg.test_pipeline if x['type'] != 'LoadMultiViewImageFromFiles'])
    processed = pipeline(data)
    if set(processed) != MODEL_KEYS:
        raise RuntimeError('UnexpectedModelInputKeys:'+str(set(processed)))
    batch = scatter(collate([processed],samples_per_gpu=1),[0])[0]
    assert batch['img'].shape == (1,6,3,384,704)
    assert batch['img'].dtype == torch.float32
    return batch


def main():
    import platform, mmcv, mmdet
    torch.manual_seed(protocol()['seed'])
    model, cfg, load = build()
    model.cuda()
    reset(model)
    # A tiny extension call is an operator test, not a whole-model forward.
    from mmcv.ops import deform_conv2d
    from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_aggregation_func
    with torch.no_grad():
        x=torch.ones(1,1,4,4,device='cuda');offset=torch.zeros(1,18,4,4,device='cuda')
        y=deform_conv2d(x,offset,torch.ones(1,1,3,3,device='cuda'),1,1)
        assert torch.isfinite(y).all()
        features=torch.ones(1,1,4,8,device='cuda')
        shape=torch.tensor([[2,2]],device='cuda',dtype=torch.int32)
        starts=torch.zeros(1,device='cuda',dtype=torch.int32)
        location=torch.full((1,2,1,2),.5,device='cuda')
        weights=torch.ones(1,2,1,1,1,device='cuda')
        aggregate=deformable_aggregation_func(features,shape,starts,location,weights)
        assert aggregate.shape==(1,2,8) and torch.isfinite(aggregate).all()
        torch.testing.assert_close(aggregate,torch.ones_like(aggregate))
    record=dict(status='StrictModelLoadPassed',load=load,python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
        gpu=torch.cuda.get_device_name(),parameters=sum(x.numel() for x in model.parameters()),
        parameter_dtypes=sorted({str(x.dtype) for x in model.parameters()}),
        buffer_dtypes=sorted({str(x.dtype) for x in model.buffers()}),
        full_model_forwards=0,optimizer_steps=0,mmcv_dcn_fp32_finite=True,native_daf_fp32_constant_field_passed=True,
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    Path('reports/m8j12_sources/model_load.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
