"""Eight actual samples through frozen official conversion and image transforms.

All per-object annotations and images remain private. No model construction.
"""
import copy
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import ObjectRangeFilter, ObjectNameFilter
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage, NormalizeMultiviewImage, PadMultiViewImage

from datasets.m8j6_selection import official_splits, select_samples, sha256


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class SelectedMetadataView:
    """Limit converter iteration while all devkit lookups retain full indexes."""
    def __init__(self, source, selected):
        self.source = source
        self.sample = selected

    def __getattr__(self, name):
        return getattr(self.source, name)


def draw_box(image, corners, projection, color):
    """nuScenes corner order; clip each edge to near plane and image bounds."""
    q = np.c_[corners, np.ones(8)] @ projection.T
    visible = False
    for i, j in EDGES:
        a, b = q[i, :3].copy(), q[j, :3].copy()
        if a[2] < .1 and b[2] < .1:
            continue
        if a[2] < .1:
            a += (b-a) * ((.1-a[2]) / (b[2]-a[2]))
        if b[2] < .1:
            b += (a-b) * ((.1-b[2]) / (a[2]-b[2]))
        u = tuple(np.clip(np.round(a[:2]/a[2]), -1000000, 1000000).astype(int))
        v = tuple(np.clip(np.round(b[:2]/b[2]), -1000000, 1000000).astype(int))
        ok, u, v = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), u, v)
        if ok:
            cv2.line(image, u, v, color, 2, cv2.LINE_AA); visible = True
    return visible


def main():
    os.umask(0o077)
    root = Path(os.environ['M8J6_ROOT']); rt = Path(os.environ['M8J6_RUNTIME']); rep = Path(os.environ['M8J6_REPORT'])
    out = root / 'prepared'; out.mkdir(exist_ok=False)
    for sub in ['rgb_input', 'supervision', 'review']:
        (out / sub).mkdir()
    cfg = json.loads(Path('configs/m8j6_protocol.json').read_text())
    nusc = NuScenes(version='v1.0-mini', dataroot=str(root / 'data'), verbose=False)
    splits = official_splits(rt / 'env/lib/python3.8/site-packages/nuscenes/utils/splits.py')
    selected = select_samples(nusc.scene, nusc.sample, splits, cfg)
    # Bound devkit methods must retain the source table/index pair. Replacing
    # .sample on a shallow NuScenes copy breaks get('sample', token).
    subset = SelectedMetadataView(nusc, selected)
    spec = importlib.util.spec_from_file_location('frozen_nuscenes_converter', rt / 'sources/PETR/tools/data_converter/nuscenes_converter.py')
    conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
    train, val = conv._fill_trainval_infos(subset, {s['scene_token'] for s in selected}, set(), False, max_sweeps=0)
    assert len(train) == 8 and len(val) == 0
    dataset = object.__new__(CustomNuScenesDataset)
    dataset.CLASSES = cfg['classes']; dataset.with_velocity = True; dataset.use_valid_flag = True
    dataset.box_mode_3d = 0; dataset.test_mode = False; dataset.modality = dict(use_camera=True); dataset.data_infos = train
    resize = ResizeCropFlipImage(dict(resize_lim=(.8,1.), final_dim=(512,1408), bot_pct_lim=(0.,0.),
        rot_lim=(0.,0.), H=900, W=1600, rand_flip=True), training=False)
    normalize = NormalizeMultiviewImage(mean=[103.530,116.280,123.675], std=[1.,1.,1.], to_rgb=False)
    frames = []; counts = Counter(); object_ids = set(); max_projection_error = 0.; max_corner_error = 0.
    for index, (sample, info) in enumerate(zip(selected, train)):
        assert sample['token'] == info['token'] and list(info['cams']) == cfg['cameras']
        inp = dataset.get_data_info(index)
        ann = dataset.get_ann_info(index)
        ann = ObjectRangeFilter(cfg['geometry']['point_cloud_range'])(ann)
        ann = ObjectNameFilter(cfg['classes'])(ann)
        gt = ann['gt_bboxes_3d'].tensor.numpy().copy(); labels = ann['gt_labels_3d'].copy()
        assert len(gt) > 0 and np.isfinite(gt).all() and (gt[:, 3:6] > 0).all()
        _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        assert [b.token for b in boxes] == sample['anns']
        trace = []; keep = []; frame_classes = Counter(); filter_counts = Counter()
        for j, (token, box) in enumerate(zip(sample['anns'], boxes)):
            raw = nusc.get('sample_annotation', token); object_ids.add(raw['instance_token'])
            source = info['gt_boxes'][j]; name = str(info['gt_names'][j]); reasons = []
            if not info['valid_flag'][j]: reasons.append('OfficialZeroLidarAndRadarPointSupport')
            if not (-51.2 < source[0] < 51.2 and -51.2 < source[1] < 51.2): reasons.append('OfficialXYCenterRange')
            if name not in cfg['classes']: reasons.append('OfficialUnsupportedTaxonomy')
            assert np.isfinite(source).all() and (source[3:6] > 0).all()
            converted = LiDARInstance3DBoxes(torch.from_numpy(source[None].astype(np.float32)), box_dim=7, origin=(.5,.5,.5))
            yaw_box = Box(box.center, box.wlh, Quaternion(axis=[0,0,1], angle=box.orientation.yaw_pitch_roll[0]))
            dist = np.linalg.norm(converted.corners[0].numpy()[:,None] - yaw_box.corners().T[None], axis=-1)
            error = float(dist.min(axis=1).max()); max_corner_error = max(max_corner_error, error)
            assert error < 1e-4
            eligible = not reasons
            if eligible: keep.append(j); frame_classes[name] += 1
            for reason in reasons: filter_counts[reason] += 1
            trace.append(dict(annotation_token=token, instance_token=raw['instance_token'],
                source_category=raw['category_name'], mapped_class=name, visibility_token=raw['visibility_token'],
                num_lidar_pts=raw['num_lidar_pts'], num_radar_pts=raw['num_radar_pts'],
                raw_global_center=raw['translation'], raw_full_wlh=raw['size'], raw_rotation=raw['rotation'],
                converted_reference_center=source[:3].tolist(), converted_full_wlh=source[3:6].tolist(),
                converted_second_yaw=float(source[6]), velocity_nan_official_zero=bool(np.isnan(info['gt_velocity'][j,0])),
                retained=eligible, filtering_reasons=reasons, corner_yaw_contract_error_m=error,
                roll_pitch_approximation_official_rad=list(map(float, box.orientation.yaw_pitch_roll[1:]))))
        assert len(keep) == len(gt)
        np.testing.assert_allclose(gt[:,:2], info['gt_boxes'][keep,:2], atol=1e-5)
        np.testing.assert_allclose(gt[:,2] + gt[:,5]/2, info['gt_boxes'][keep,2], atol=1e-5)
        assert labels.tolist() == [cfg['classes'].index(str(info['gt_names'][j])) for j in keep]
        images = []; hashes = []; panels = []; frame_proj = []; views = []
        for ci, cam in enumerate(cfg['cameras']):
            cam_info = info['cams'][cam]; path = Path(cam_info['data_path'])
            img = cv2.imread(str(path)); assert img.shape == (900,1600,3)
            images.append(img); hashes.append(dict(camera=cam, path=str(path), sha256=sha256(path),
                sample_data_token=cam_info['sample_data_token'], timestamp_us=cam_info['timestamp']))
            source_view = img.copy(); final_view = img.copy(); vv = Counter()
            _, cam_boxes, K = nusc.get_sample_data(sample['data'][cam], selected_anntokens=sample['anns'], box_vis_level=0)
            by_token = {b.token:b for b in cam_boxes}
            ext = inp['extrinsics'][ci].T
            for j, box in enumerate(boxes):
                reference_corners = box.corners().T
                transformed = (np.c_[reference_corners,np.ones(8)] @ ext.T)[:,:3]
                if box.token in by_token:
                    residual = float(np.max(abs(transformed - by_token[box.token].corners().T)))
                    assert residual < 1e-8; frame_proj.append(residual)
                color = (0,220,0) if j in keep else (0,150,255)
                visible = draw_box(source_view, reference_corners, inp['lidar2img'][ci], color)
                if visible: vv['raw_projected'] += 1
                if j in keep:
                    v = draw_box(final_view, reference_corners, inp['lidar2img'][ci], color)
                    if v: vv['retained_projected'] += 1
            for title, target in [('RAW all: green kept / orange filtered', source_view), ('TRAIN GT', final_view)]:
                cv2.putText(target, cam + ' ' + title, (20,35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255),2)
            cv2.imwrite(str(out/'review'/f'{index:02}_{cam}_raw.jpg'),source_view)
            cv2.imwrite(str(out/'review'/f'{index:02}_{cam}_train.jpg'),final_view)
            panels.append(cv2.resize(source_view,(640,360))); views.append(dict(camera=cam,**vv))
        mosaic = np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
        cv2.imwrite(str(out/'review'/f'{index:02}_six.jpg'),mosaic)
        bev = np.zeros((900,900,3),np.uint8)
        cv2.line(bev,(450,0),(450,900),(70,70,70),1);cv2.line(bev,(0,450),(900,450),(70,70,70),1)
        for j, box in enumerate(boxes):
            pts=box.bottom_corners()[:2].T; pix=np.stack([450+pts[:,0]*8,450-pts[:,1]*8],axis=1).round().astype(np.int32)
            cv2.polylines(bev,[pix],True,(0,220,0) if j in keep else (0,150,255),1)
        cv2.putText(bev,sample['scene_name']+' '+sample['token'][:8],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.imwrite(str(out/'review'/f'{index:02}_bev.png'),bev)
        data=resize(dict(img=[im.astype(np.float32) for im in images],intrinsics=copy.deepcopy(inp['intrinsics']),extrinsics=copy.deepcopy(inp['extrinsics'])))
        A=np.array([[.88,0,0],[0,.88,-280],[0,0,1]])
        for old,new,ext,proj in zip(inp['intrinsics'],data['intrinsics'],data['extrinsics'],data['lidar2img']):
            np.testing.assert_allclose(new[:3,:3],A @ old[:3,:3],atol=1e-4)
            np.testing.assert_allclose(new @ ext.T,proj,atol=1e-8)
        data=PadMultiViewImage(size_divisor=32)(normalize(data))
        tensor=np.stack(data['img']).transpose(0,3,1,2).copy()[None]
        assert tensor.shape==(1,6,3,512,1408) and tensor.dtype==np.float32 and np.isfinite(tensor).all()
        np.savez(out/'rgb_input'/f'{index:02}.npz',images=tensor,lidar2img=np.asarray(data['lidar2img']),
                 camera_timestamp_us=np.asarray([c['timestamp_us'] for c in hashes],dtype=np.int64))
        np.savez(out/'supervision'/f'{index:02}.npz',boxes=gt,labels=labels)
        save(out/'supervision'/f'{index:02}_trace.json',trace)
        max_projection_error=max(max_projection_error,max(frame_proj,default=0.))
        record=dict(index=index,sample_token=sample['token'],scene_name=sample['scene_name'],scene_token=sample['scene_token'],
            timestamp_us=sample['timestamp'],camera_files=hashes,raw_annotations=len(trace),retained_gt=len(gt),
            retained_classes=dict(frame_classes),filter_reason_counts=dict(filter_counts),camera_views=views,
            camera_time_offset_us=[c['timestamp_us']-sample['timestamp'] for c in hashes],
            velocity_zero_fallback=sum(t['velocity_nan_official_zero'] and t['retained'] for t in trace),
            projection_coordinate_residual_m=max(frame_proj,default=0.),
            input_sha256=sha256(out/'rgb_input'/f'{index:02}.npz'),supervision_sha256=sha256(out/'supervision'/f'{index:02}.npz'),
            unresolved_supervision_anomalies=[],visual_review='PendingActualInspection')
        frames.append(record);counts.update(frame_classes)
        print(index,sample['scene_name'],len(trace),len(gt),dict(filter_counts),flush=True)
    save(out/'manifest.json',dict(frames=frames,protocol_sha256=sha256('configs/m8j6_protocol.json'),
        six_camera_order=cfg['cameras'],training_classes=cfg['classes'],data_source='nuScenesOfficialMini_UserDownloaded',
        model_input_fields=['images','lidar2img','camera_timestamp_us'],GT_separate=True,point_cloud_payload_reads=0))
    links=''.join(f'<h2>Frame {i:02}</h2><img width="1500" src="{i:02}_six.jpg"><img width="650" src="{i:02}_bev.png">' +
        ''.join(f'<p>{c}: <a href="{i:02}_{c}_raw.jpg">all raw GT</a> | <a href="{i:02}_{c}_train.jpg">training GT</a></p>' for c in cfg['cameras']) for i in range(8))
    (out/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>M8-J6 private all-object review</title><body>'+links)
    save(rep/'result.json',dict(status='EightFrameAutomaticContractPassed_VisualCheckPending',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),frames=frames,
        totals=dict(raw=sum(f['raw_annotations'] for f in frames),retained=sum(f['retained_gt'] for f in frames),
                    unique_raw_instances=len(object_ids),retained_classes=dict(counts)),
        max_yaw_box_corner_error_m=max_corner_error,max_camera_coordinate_error_m=max_projection_error,
        private_manifest_sha256=sha256(out/'manifest.json'),raw_annotation_trace='PrivateOnly',
        point_cloud_payload_reads=0,model_input_isolation='SeparateNPZ_Whitelist',actual_human_signoff=False))


if __name__ == '__main__':
    main()
