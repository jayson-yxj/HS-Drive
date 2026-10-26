"""Pure six-camera numeric input to existing PETR test preprocessing."""
import numpy as np

CAMERA_ORDER=['front','front_left','front_right','rear_left','rear_right','rear']
A=np.array([[.88,0,0],[0,.88,-280],[0,0,1]],np.float64)


def validate_arrays(images,ks,ts):
    if images.shape!=(6,900,1600,3) or images.dtype!=np.uint8:
        raise ValueError('Expected six actual full-resolution RGB images')
    if ks.shape!=(6,3,3) or ts.shape!=(6,4,4):
        raise ValueError('Calibration shape mismatch')
    if not np.isfinite(ks).all() or not np.isfinite(ts).all():
        raise ValueError('Nonfinite calibration')
    for k,t in zip(ks,ts):
        if not np.allclose(k[2],[0,0,1]) or min(k[0,0],k[1,1])<=0:
            raise ValueError('Invalid intrinsic')
        if not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('Invalid extrinsic')


def prepare(images,ks,ts):
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage,NormalizeMultiviewImage,PadMultiViewImage
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    validate_arrays(images,ks,ts)
    intr=[]
    for k in ks:
        v=np.eye(4);v[:3,:3]=k;intr.append(v)
    tr=ResizeCropFlipImage(dict(H=900,W=1600,final_dim=(512,1408),resize_lim=(.8,1.),
        bot_pct_lim=(0.,0.),rot_lim=(0.,0.),rand_flip=True),training=False)
    aug=tr._sample_augmentation()
    if aug!=(.88,(1408,792),(0,280,1408,792),False,0):
        raise ValueError('Unexpected official test augmentation')
    data=tr(dict(img=[x[:,:,::-1].copy() for x in images],intrinsics=intr,
                 extrinsics=[np.linalg.inv(t).T for t in ts]))
    residual=[]
    for k,t,ka,proj in zip(ks,ts,data['intrinsics'],data['lidar2img']):
        assert np.allclose(ka[:3,:3],A@k,atol=1e-4)
        opt=np.array([[0,0,10,1],[1,-1,20,1],[-2,1,30,1]],float)
        ego=opt@t.T;q=ego@proj.T
        expect=opt[:,:3]@(A@k).T
        residual.append(float(np.max(abs(q[:,:2]/q[:,2:3]-expect[:,:2]/expect[:,2:3]))))
    data=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False)(data)
    data=PadMultiViewImage(size_divisor=32)(data)
    cpu=np.stack(data['img']).transpose(0,3,1,2).copy()[None]
    assert cpu.shape==(1,6,3,512,1408) and cpu.dtype==np.float32 and np.isfinite(cpu).all()
    meta=dict(img_shape=[(512,1408,3)]*6,pad_shape=[(512,1408,3)]*6,ori_shape=[(900,1600,3)]*6,
        lidar2img=data['lidar2img'],box_type_3d=LiDARInstance3DBoxes,scale_factor=1.,flip=False)
    audit=dict(camera_order=CAMERA_ORDER,source_frame=None,timestamp_s=None,temporal_valid=False,
        A=A.tolist(),resize_dims=[1408,792],crop=[0,280,1408,792],
        intrinsic_aug=[k.tolist() for k in data['intrinsics']],
        lidar2img=[p.tolist() for p in data['lidar2img']],projection_residual_px=max(residual),
        actual_input_size=[900,1600],model_input_size=[512,1408],GT_fields=[],
        preprocessing='Unmodified official test transform; changed data H/W explicitly; no HS calibration substitution')
    return cpu,meta,audit
