"""Candidate's official test transforms, metadata-only pose adapter; no GT read."""
import copy,hashlib,json,os,time,sys
from pathlib import Path
import cv2,mmcv,numpy as np
from datasets.m8j11_contract import *
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipRotImage,NormalizeMultiviewImage,PadMultiViewImage

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if Path(path).name in ('sample_annotation.json','instance.json','visibility.json','category.json') or path.endswith(('.pcd','.bin')):
                raise PermissionError('PreparationMetadataRGBOnly_NoGTOrPointPayload')
    sys.addaudithook(guard)
    os.umask(0o077);c=json.loads(Path('configs/m8j11_protocol.json').read_text());base=json.loads(Path('configs/m8j9_retry1_protocol.json').read_text())
    root=Path(c['private_root'])/'prepared';root.mkdir(parents=True,exist_ok=False)
    fixed=json.loads(Path(c['eval_manifest']).read_text());t=read_metadata(base['dataset_root']);check_sequence(fixed['frames'],t)
    config=mmcv.Config.fromfile(str(Path(c['runtime_root'])/'runtime-source'/c['config']))
    resize=ResizeCropFlipRotImage(config.ida_aug_conf,training=False);norm=NormalizeMultiviewImage(**config.img_norm_cfg);pad=PadMultiViewImage(size_divisor=32)
    assert resize._sample_augmentation()==(.44,(704,396),(0,140,704,396),False,0)
    ds=object.__new__(CustomNuScenesDataset);ds.test_mode=True;ds.modality=dict(use_camera=True)
    ds.data_infos=[info_from_metadata(f,t,base['dataset_root']) for f in fixed['frames']]
    rows=[];max_projection=max_pose=0.
    for f in fixed['frames']:
        t0=time.perf_counter();i=f['index'];inp=ds.get_data_info(i);assert 'ann_info' not in inp
        data=dict(img=[cv2.imread(p).astype(np.float32) for p in inp['img_filename']],intrinsics=copy.deepcopy(inp['intrinsics']),extrinsics=copy.deepcopy(inp['extrinsics']))
        assert all(im.shape==(900,1600,3) for im in data['img'])
        data=pad(norm(resize(data)));preprocessing_ms=(time.perf_counter()-t0)*1000;A=np.array([[.44,0,0],[0,.44,-140],[0,0,1]])
        for j in range(6):
            np.testing.assert_allclose(data['intrinsics'][j][:3,:3],A@inp['intrinsics'][j][:3,:3],atol=1e-5)
            np.testing.assert_allclose(data['lidar2img'][j],data['intrinsics'][j]@data['extrinsics'][j],atol=1e-8)
            # Independent calibrated-camera global path, evaluated at each camera's real pose.
            s=t['sample'][f['sample_token']];sd=t['sample_data'][s['data'][CAMERAS[j]]]
            expected=np.linalg.inv(sensor_global(sd,t))@inp['ego_pose']
            max_projection=max(max_projection,float(np.max(abs(expected-data['extrinsics'][j]))))
        max_pose=max(max_pose,float(np.max(abs(inp['ego_pose_inv']@inp['ego_pose']-np.eye(4)))))
        values=dict(images=np.stack(data['img']).transpose(0,3,1,2).copy()[None])
        for k in INPUT_FIELDS-{'images'}:
            values[k]=np.asarray(data[k] if k in ('lidar2img','intrinsics','extrinsics') else inp[k],dtype=np.float64 if k in ('timestamp','img_timestamp') else np.float32)
        assert values['images'].shape==(1,6,3,256,704);assert set(values)==INPUT_FIELDS
        p=root/f'{i:02}.npz';np.savez(p,**values)
        rows.append(dict(f,path=str(p),sha256=sha(p),raw_image_preprocessing_ms=preprocessing_ms,camera_timestamp_us=[t['sample_data'][t['sample'][f['sample_token']]['data'][cam]]['timestamp'] for cam in CAMERAS],
            camera_image_sha256=[sha(p) for p in inp['img_filename']]))
    assert max_projection<1e-8 and max_pose<1e-8
    record=dict(status='CandidateInputPrepared',frames=rows,input_fields=sorted(INPUT_FIELDS),cameras=CAMERAS,
        image_shape=[1,6,3,256,704],normalize=dict(config.img_norm_cfg),timestamp_dtype='float64_seconds_from_real_microseconds',
        ego_pose='LIDAR_TOP reference to global, from calibrated_sensor and ego_pose, no pointcloud payload',
        max_extrinsic_error=max_projection,max_pose_inverse_error=max_pose,GT_accessed=False,pointcloud_payload_reads=0,
        source_manifest_sha256=sha(c['eval_manifest']))
    (root/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    Path('reports/m8j11_sources/input_preparation.json').write_text(json.dumps({k:v for k,v in record.items() if k!='frames'},indent=2)+'\n')
    print('Prepared',len(rows),'frames')
if __name__=='__main__':main()
