"""Minimal offline calibration checks. Run only after original GPU forward passes."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sys
import os
sys.path.insert(0,os.environ['PETR_SOURCE'])
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage, NormalizeMultiviewImage
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox,denormalize_bbox

ORDER=['front','front_left','front_right','rear_left','rear_right','rear']

def main():
    assert os.environ.get('M8A_ISOLATED')=='1'
    p=json.loads(Path('/input/shadow.json').read_text());assert p['camera_order']==ORDER
    assert p['image_size']==[384,640]
    assert all(c['frame_id']==p['frame_id'] and c['timestamp_s']==p['timestamp_s'] for c in p['cameras'])
    # Ignore ego_pose completely; no actor/depth/instance/GT fields passed to any model.
    bgr=[np.array(Image.open('/input/'+name+'.png').convert('RGB'))[...,::-1].copy() for name in ORDER]
    intr=[np.eye(4) for _ in ORDER]
    for k,mat in zip(p['intrinsics'],intr):mat[:3,:3]=k
    c2e=[np.array(t) for t in p['cam_to_ego']]
    ext=[np.linalg.inv(t).T for t in c2e]
    aug=dict(H=384,W=640,final_dim=(512,1408),resize_lim=(.8,1.),bot_pct_lim=(0.,0.),rot_lim=(0.,0.),rand_flip=True)
    transform=ResizeCropFlipImage(aug,training=False)
    scale,dims,crop,flip,rot=transform._sample_augmentation()
    A=np.array([[scale,0,-crop[0]],[0,scale,-crop[1]],[0,0,1.]])
    res=transform(dict(img=bgr,intrinsics=intr,extrinsics=ext))
    assert scale==2.2 and dims==(1408,844) and crop==(0,332,1408,844) and not flip and rot==0
    checks=[]
    for index,name in enumerate(ORDER):
        K=np.array(p['intrinsics'][index]);T=c2e[index];Knew=res['intrinsics'][index][:3,:3]
        assert np.allclose(Knew,A@K,atol=1e-4)
        assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=2e-6)
        assert abs(np.linalg.det(T[:3,:3])-1)<2e-6
        xyz=np.array([[-2,-1,5,1],[0,0,10,1],[2,2,20,1],[1,-1,-5,1]],dtype=float)
        ego=xyz@T.T;proj=ego@res['lidar2img'][index].T
        expected=xyz[:,:3]@(A@K).T
        assert np.max(abs(proj[:,:3]-expected))<1e-3
        assert np.all(proj[:3,2]>0) and proj[3,2]<0
        restored=proj@np.linalg.inv(res['lidar2img'][index]).T
        err=float(np.max(abs(restored-ego)));assert err<1e-10
        checks.append(dict(camera=name,roundtrip_xyz_max_m=err,new_K=Knew.tolist(),T_cam_to_ego=T.tolist(),lidar2img=res['lidar2img'][index].tolist()))
    # Official old mmdet3d LiDAR box: w,l,h; clockwise yaw from -y, bottom center.
    # CARLA left-handed yaw -> HS counterclockwise yaw = -yaw_carla.
    center=np.array([12.,3.,1.7]);length,width,height=4.6,1.8,1.5
    yaw_carla=.37;hs_yaw=-yaw_carla;petr_yaw=-hs_yaw-math.pi/2
    box=LiDARInstance3DBoxes(torch.tensor([[*center,width,length,height,petr_yaw]],dtype=torch.float32),origin=(.5,.5,.5))
    actual=box.corners[0].numpy()
    R=np.array([[np.cos(hs_yaw),-np.sin(hs_yaw),0],[np.sin(hs_yaw),np.cos(hs_yaw),0],[0,0,1.]])
    known=np.array([[x*length/2,y*width/2,z*height/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])@R.T+center
    error=float(np.max(np.min(np.linalg.norm(actual[:,None]-known[None],axis=-1),axis=1)))
    assert error<3e-6,error
    assert np.allclose(box.gravity_center.numpy()[0],center,atol=1e-6)
    assert abs(box.tensor[0,2].item()-(center[2]-height/2))<1e-6
    normalized=normalize_bbox(box.tensor,None);roundtrip=denormalize_bbox(normalized,None)
    assert torch.allclose(roundtrip,box.tensor,atol=1e-6)
    # Non-square box projection and inverse use measured front calibration.
    homo=np.c_[actual,np.ones(8)];proj=homo@res['lidar2img'][0].T
    recovered=proj@np.linalg.inv(res['lidar2img'][0]).T
    box_error=float(np.max(abs(recovered-homo)));assert box_error<1e-10
    normalized_images=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False)(res)
    assert np.allclose(normalized_images['img'][0][0,0],res['img'][0][0,0]) # same dict mutated
    # Independent RGB channel sentinel, through the official normalization code.
    rgb=np.array([[[10.,30.,90.]]],dtype=np.float32)
    sentinel=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False)(dict(img=[rgb[...,::-1].copy()]))['img'][0]
    assert np.allclose(sentinel[0,0],[90-103.530,30-116.280,10-123.675],atol=1e-5)
    result=dict(status='Passed_BasicReadOnlyContract',kind='OfflineM7AShadowCalibrationAndSyntheticBoxTests',sample_id=p['sample_id'],
        frame_id=p['frame_id'],timestamp_s=p['timestamp_s'],camera_order=ORDER,
        source_image_shape=[384,640,3],output_image_shape=[512,1408,3],RGB_to_BGR=True,official_to_rgb=False,
        augmentation=dict(scale=scale,resized_dims=dims,crop=crop,flip=flip,rotation_deg=rot,matrix=A.tolist(),
            integer_resize_height_rounding_pixels=.8,note='Official nominal scale convention; image sampling has subpixel rounding; substantial upper-image crop, not quality validated'),
        camera_checks=checks,HS_frame='x_forward_y_left_z_up',camera_frame='x_right_y_down_z_forward',
        model_reference_frame='HS ego used as right-handed pseudo-LiDAR frame; not claiming identical nuScenes sensor mounting',
        old_framework_box=dict(center='bottom_center; head internally gravity center then z-=h/2',dims='width,length,height',
            yaw_formula='yaw_old_lidar = -yaw_HS - pi/2; yaw_HS=-yaw_CARLA',non_square_corner_max_m=error,
            projection_roundtrip_max_m=box_error),GT_model_input=False,carla_model_forward=False,
        limitations=['No full adapter','No zero-shot or detection quality evaluation','No live synchronization evidence','No control or planning authorization'])
    Path('/output/camera_contract.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
