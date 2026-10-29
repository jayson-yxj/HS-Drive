"""End-to-end local RGB read/decode/preprocess/calibration/H2D/model/D2H timing."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from datasets.m8j11_contract import read_metadata,info_from_metadata
from training.m8j6_retry_state import snapshot,assert_exact
from training.m8j7_metrics import distribution

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['A','StreamPETR'],required=True);a=ap.parse_args()
    c=json.loads(Path('configs/m8j11_protocol.json').read_text());j=json.loads(Path('configs/m8j9_retry1_protocol.json').read_text());rt=Path(c['runtime_root']);private=Path(c['private_root'])
    mf=json.loads(Path(j['manifest']).read_text());frames=mf['frames'];tables=read_metadata(j['dataset_root'])
    torch.set_num_threads(2);torch.manual_seed(20261029);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.model=='StreamPETR':
        import mmcv
        import projects.mmdet3d_plugin
        from mmdet3d.models import build_model
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipRotImage as Resize,NormalizeMultiviewImage,PadMultiViewImage
        from datasets.m8j11_contract import to_legacy_boxes
        conf=mmcv.Config.fromfile(str(rt/'runtime-source'/c['config']));conf.model.img_backbone.pretrained=None
        att=conf.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1];att.type='PETRMultiheadAttention';att.fp16=False
        model=build_model(conf.model,train_cfg=conf.get('train_cfg'),test_cfg=conf.get('test_cfg'))
        cp=rt/'artifacts/stream_petr_r50_flash_704_bs2_seq_90e.pth';assert hashlib.sha256(cp.read_bytes()).hexdigest()==c['initial_asset_sha256']
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict'],strict=True);model.float().cuda().eval()
        ds=object.__new__(CustomNuScenesDataset);ds.test_mode=True;ds.modality={'use_camera':True};ds.data_infos=[info_from_metadata(f,tables,j['dataset_root']) for f in frames]
        resize=Resize(conf.ida_aug_conf,training=False);norm=NormalizeMultiviewImage(**conf.img_norm_cfg);H,W=256,704
        cache=private/'full';inputs=private/'prepared';source_manifest=json.loads((inputs/'manifest.json').read_text())
        model.prev_scene_token=None;model.pts_bbox_head.reset_memory()
    else:
        from tools import m8j6_gpu_actual as old
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage as Resize,NormalizeMultiviewImage,PadMultiViewImage
        LiDARInstance3DBoxes=old.LiDARInstance3DBoxes
        cfg=json.loads(Path('configs/m8j6_protocol.json').read_text());model=old.build(Path(j['runtime']),cfg,Path(j['A']['path']))
        resize=Resize(dict(resize_lim=(.8,1.),final_dim=(512,1408),bot_pct_lim=(0.,0.),rot_lim=(0.,0.),H=900,W=1600,rand_flip=True),training=False)
        norm=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False);H,W=512,1408
        cache=Path(j['cache_root']);inputs=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared/rgb_input')
        source_manifest=json.loads((inputs.parent/'manifest.json').read_text())
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    initial=snapshot(model.state_dict());pad=PadMultiViewImage(size_divisor=32);rows=[]
    for f in frames:
        i=f['index'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        if a.model=='StreamPETR':
            inp=ds.get_data_info(i);data=dict(img=[cv2.imread(p).astype(np.float32) for p in inp['img_filename']],intrinsics=copy.deepcopy(inp['intrinsics']),extrinsics=copy.deepcopy(inp['extrinsics']))
        else:
            info=info_from_metadata(f,tables,j['dataset_root']);data=dict(img=[],intrinsics=[],extrinsics=[])
            for name,cam in info['cams'].items():
                data['img'].append(cv2.imread(cam['data_path']).astype(np.float32))
                k=np.eye(4);k[:3,:3]=cam['cam_intrinsic'];ext=np.eye(4);ext[:3,:3]=cam['sensor2lidar_rotation'];ext[:3,3]=cam['sensor2lidar_translation']
                data['intrinsics'].append(k);data['extrinsics'].append(np.linalg.inv(ext).T)
        data=pad(norm(resize(data)));cpuimg=np.stack(data['img']).transpose(0,3,1,2).copy()[None];img=torch.from_numpy(cpuimg).cuda()
        meta=[dict(scene_token=f['scene_token'],lidar2img=data['lidar2img'],img_timestamp=[v/1e6 for v in source_manifest['frames'][i].get('camera_timestamp_us',[])],img_shape=[(H,W,3)]*6,pad_shape=[(H,W,3)]*6,ori_shape=[(900,1600,3)]*6,box_type_3d=LiDARInstance3DBoxes,scale_factor=1.,flip=False)]
        tensors={}
        if a.model=='StreamPETR':
            for k in ('lidar2img','intrinsics','extrinsics','timestamp','img_timestamp','ego_pose','ego_pose_inv'):
                val=data[k] if k in data else inp[k];tensors[k]=torch.from_numpy(np.asarray(val,dtype=np.float64 if k in ('timestamp','img_timestamp') else np.float32)).cuda().unsqueeze(0)
        torch.cuda.synchronize();prep=time.perf_counter();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            if a.model=='StreamPETR':
                pred=model.simple_test(meta,img=img,**tensors)[0]['pts_bbox'];b,s,l=pred['boxes_3d'],pred['scores_3d'],pred['labels_3d']
            else:
                feats=model.extract_feat(img,meta);raw=model.pts_bbox_head(feats,meta);b,s,l=model.pts_bbox_head.get_bboxes(raw,meta)[0]
        e1.record();e1.synchronize();boxes=b.tensor.cpu().numpy();scores=s.cpu().numpy();labels=l.cpu().numpy();end=time.perf_counter()
        if a.model=='StreamPETR':boxes=to_legacy_boxes(boxes)
        with np.load(cache/f'{i:02}.npz',allow_pickle=False) as z:
            for k,v in [('boxes',boxes),('scores',scores),('labels',labels)]:np.testing.assert_allclose(v,z[k],atol=1e-5,rtol=1e-5)
        with np.load(inputs/f'{i:02}.npz',allow_pickle=False) as z:np.testing.assert_array_equal(cpuimg,z['images'])
        assert_exact(initial,model.state_dict(),'EndToEndReadOnly')
        rows.append(dict(index=i,cold_scene=i in (0,40),raw_to_cpu_prediction_ms=(end-t)*1000,RGB_decode_preprocess_H2D_ms=(prep-t)*1000,gpu_ms=e0.elapsed_time(e1),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(a.model,i,rows[-1]['raw_to_cpu_prediction_ms'],flush=True)
        del b,s,l,img,tensors,data
    result=dict(status='RawRGBTimingAndCacheEquivalencePassed',model=a.model,frames=rows,optimizer_steps=0,backward_calls=0,
        metadata_loaded_before_timing=True,boundary='Per-frame RGB file read+JPEG decode+official geometry/preprocessing+H2D+model/native decode+D2H; no GT, cache comparison/state audits/serialization outside timer',summary={})
    for k,rr in [('all',rows),('cold_scene',[r for r in rows if r['cold_scene']]),('subsequent',[r for r in rows if not r['cold_scene']])]:
        result['summary'][k]={key:distribution([r[key] for r in rr]) for key in ('raw_to_cpu_prediction_ms','RGB_decode_preprocess_H2D_ms','gpu_ms','peak_allocated_bytes','peak_reserved_bytes')}
    with Path(f'reports/m8j11_sources/{a.model}_raw_timing.json').open('x') as out:json.dump(result,out,indent=2)
if __name__=='__main__':main()
