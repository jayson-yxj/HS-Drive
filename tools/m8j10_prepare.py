"""Local intersection-train official adaptation; same frozen converter and checks. No model, training or downloads."""
from collections import Counter
from datetime import datetime,timezone
import copy
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import ObjectRangeFilter,ObjectNameFilter
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage,NormalizeMultiviewImage,PadMultiViewImage
from tools.m8j6_prepare_actual import SelectedMetadataView,save
from datasets.m8j6_selection import sha256


def main():
    os.umask(0o077)
    cfg=json.loads(Path('configs/m8j6_protocol.json').read_text())
    protocol=json.loads(Path('configs/m8j10_protocol.json').read_text())
    fixed=json.loads(Path(protocol['manifest']).read_text())
    assert sha256(protocol['manifest'])==protocol['manifest_sha256']
    rt=Path(os.environ['M8J6_RUNTIME']);out=Path(protocol['prepared_root'])
    out.mkdir(parents=True,exist_ok=False)
    for sub in ('rgb_input','supervision'):(out/sub).mkdir()
    nusc=NuScenes(version='v1.0-mini',dataroot=protocol['dataset_root'],verbose=False)
    selected=[dict(nusc.get('sample',r['sample_token']),scene_name=r['scene_name']) for r in fixed['frames']]
    subset=SelectedMetadataView(nusc,selected)
    spec=importlib.util.spec_from_file_location('frozen_converter',rt/'sources/PETR/tools/data_converter/nuscenes_converter.py')
    conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)
    infos,val=conv._fill_trainval_infos(subset,{s['scene_token'] for s in selected},set(),False,max_sweeps=0)
    assert not val and len(infos)==len(selected)==len(fixed['frames'])
    dataset=object.__new__(CustomNuScenesDataset)
    dataset.CLASSES=cfg['classes'];dataset.with_velocity=True;dataset.use_valid_flag=True
    dataset.box_mode_3d=0;dataset.test_mode=False;dataset.modality=dict(use_camera=True);dataset.data_infos=infos
    resize=ResizeCropFlipImage(dict(resize_lim=(.8,1.),final_dim=(512,1408),bot_pct_lim=(0.,0.),
        rot_lim=(0.,0.),H=900,W=1600,rand_flip=True),training=False)
    norm=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False)
    counts=Counter();filters=Counter();frames=[];instances=set();retained_instances=set()
    max_corner=max_camera=0.
    for index,(sample,info) in enumerate(zip(selected,infos)):
        assert info['token']==sample['token'] and list(info['cams'])==cfg['cameras']
        inp=dataset.get_data_info(index);ann=dataset.get_ann_info(index)
        ann=ObjectRangeFilter(cfg['geometry']['point_cloud_range'])(ann)
        ann=ObjectNameFilter(cfg['classes'])(ann)
        gt=ann['gt_bboxes_3d'].tensor.numpy().copy();labels=ann['gt_labels_3d'].copy()
        assert np.isfinite(gt).all() and (gt[:,3:6]>0).all()
        _,boxes,_=nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        assert [b.token for b in boxes]==sample['anns']
        trace=[];keep=[];fc=Counter();ff=Counter()
        for j,(token,box) in enumerate(zip(sample['anns'],boxes)):
            raw=nusc.get('sample_annotation',token);instances.add(raw['instance_token'])
            source=info['gt_boxes'][j];name=str(info['gt_names'][j]);reasons=[]
            if not info['valid_flag'][j]:reasons.append('OfficialZeroLidarAndRadarPointSupport')
            if not (-51.2<source[0]<51.2 and -51.2<source[1]<51.2):reasons.append('OfficialXYCenterRange')
            if name not in cfg['classes']:reasons.append('OfficialUnsupportedTaxonomy')
            converted=LiDARInstance3DBoxes(torch.from_numpy(source[None].astype(np.float32)),box_dim=7,origin=(.5,.5,.5))
            yaw_box=Box(box.center,box.wlh,Quaternion(axis=[0,0,1],angle=box.orientation.yaw_pitch_roll[0]))
            delta=np.linalg.norm(converted.corners[0].numpy()[:,None]-yaw_box.corners().T[None],axis=-1)
            err=float(delta.min(axis=1).max());assert err<1e-4;max_corner=max(max_corner,err)
            if not reasons:
                keep.append(j);fc[name]+=1;retained_instances.add(raw['instance_token'])
            ff.update(reasons)
            trace.append(dict(annotation_token=token,instance_token=raw['instance_token'],retained=not reasons,
                raw_global_center=raw['translation'],raw_full_wlh=raw['size'],raw_rotation=raw['rotation'],
                mapped_class=name,visibility_token=raw['visibility_token'],
                converted_reference_center=source[:3].tolist(),converted_full_wlh=source[3:6].tolist(),
                converted_second_yaw=float(source[6]),filtering_reasons=reasons))
        assert len(keep)==len(gt)
        np.testing.assert_allclose(gt[:,:2],info['gt_boxes'][keep,:2],atol=1e-5)
        np.testing.assert_allclose(gt[:,2]+gt[:,5]/2,info['gt_boxes'][keep,2],atol=1e-5)
        assert labels.tolist()==[cfg['classes'].index(str(info['gt_names'][j])) for j in keep]
        images=[];cameras=[]
        for ci,camera in enumerate(cfg['cameras']):
            c=info['cams'][camera];path=Path(c['data_path']);im=cv2.imread(str(path));assert im.shape==(900,1600,3)
            images.append(im)
            cameras.append(dict(camera=camera,path=str(path),sha256=sha256(path),
                sample_data_token=c['sample_data_token'],timestamp_us=c['timestamp']))
            _,cam_boxes,_=nusc.get_sample_data(sample['data'][camera],selected_anntokens=sample['anns'],box_vis_level=0)
            by_token={b.token:b for b in cam_boxes}
            for box in boxes:
                if box.token not in by_token:continue
                transformed=(np.c_[box.corners().T,np.ones(8)]@inp['extrinsics'][ci].T.T)[:,:3]
                err=float(np.max(abs(transformed-by_token[box.token].corners().T)))
                assert err<1e-8;max_camera=max(max_camera,err)
        data=resize(dict(img=[im.astype(np.float32) for im in images],intrinsics=copy.deepcopy(inp['intrinsics']),extrinsics=copy.deepcopy(inp['extrinsics'])))
        A=np.array([[.88,0,0],[0,.88,-280],[0,0,1]])
        for old,new,ext,proj in zip(inp['intrinsics'],data['intrinsics'],data['extrinsics'],data['lidar2img']):
            np.testing.assert_allclose(new[:3,:3],A@old[:3,:3],atol=1e-4)
            np.testing.assert_allclose(new@ext.T,proj,atol=1e-8)
        data=PadMultiViewImage(size_divisor=32)(norm(data))
        images=np.stack(data['img']).transpose(0,3,1,2).copy()[None]
        assert images.shape==(1,6,3,512,1408) and images.dtype==np.float32
        np.savez(out/'rgb_input'/f'{index:02}.npz',images=images,lidar2img=np.asarray(data['lidar2img']),
            camera_timestamp_us=np.asarray([c['timestamp_us'] for c in cameras],dtype=np.int64))
        np.savez(out/'supervision'/f'{index:02}.npz',boxes=gt,labels=labels)
        save(out/'supervision'/f'{index:02}_trace.json',trace)
        frame=dict(index=index,sample_token=sample['token'],scene_name=sample['scene_name'],scene_token=sample['scene_token'],
            timestamp_us=sample['timestamp'],camera_files=cameras,raw_annotations=len(trace),retained_gt=len(gt),
            retained_classes=dict(fc),filter_reason_counts=dict(ff),input_sha256=sha256(out/'rgb_input'/f'{index:02}.npz'),
            supervision_sha256=sha256(out/'supervision'/f'{index:02}.npz'),unresolved_supervision_anomalies=[])
        frames.append(frame);counts.update(fc);filters.update(ff)
        print(index,sample['scene_name'],len(gt),flush=True)
    assert all(f['retained_gt']>0 for f in frames), 'ConcreteZeroPositiveFrameNeedsExplicitHandling'
    oldpre=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    oldmf=json.loads((oldpre/'manifest.json').read_text());by_token={f['sample_token']:f for f in frames}
    for old in oldmf['frames']:
        new=by_token[old['sample_token']]
        assert old['retained_gt']==new['retained_gt']
        for sub in ('rgb_input','supervision'):
            with np.load(oldpre/sub/f'{old["index"]:02}.npz') as a,np.load(out/sub/f'{new["index"]:02}.npz') as b:
                assert set(a.files)==set(b.files)
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    save(out/'manifest.json',dict(frames=frames,six_camera_order=cfg['cameras'],training_classes=cfg['classes'],
        model_input_fields=['images','lidar2img','camera_timestamp_us'],GT_separate=True,point_cloud_payload_reads=0,
        protocol_sha256=sha256('configs/m8j6_protocol.json'),selection_manifest_sha256=protocol['manifest_sha256']))
    save(Path(os.environ['M8J10_REPORT'])/'result.json',dict(status='PreparedAllIntersectionTrain',
        created_at_utc=datetime.now(timezone.utc).isoformat(),frames=frames,
        totals=dict(raw=sum(f['raw_annotations'] for f in frames),retained=sum(f['retained_gt'] for f in frames),
            unique_raw_instances=len(instances),unique_retained_instances=len(retained_instances),classes=dict(counts),filters=dict(filters)),
        max_corner_error_m=max_corner,max_camera_transform_error_m=max_camera,
        manifest_sha256=sha256(out/'manifest.json'),optimizer_steps=0,raw_data_unchanged=True))


if __name__=='__main__':main()
