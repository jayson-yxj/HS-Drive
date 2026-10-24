import copy,json,unittest
from pathlib import Path
import numpy as np
from datasets.m8b_petr_adapter import PETRCarlaOfflineAdapter,old_boxes_to_hs,corners_hs,A,ORDER,ANGLES

class Contract(unittest.TestCase):
    def setUp(self):
        self.p=json.loads(Path('/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot/rgb_bundle/s000/packet.json').read_text())
        self.adapter=PETRCarlaOfflineAdapter('/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot/rgb_bundle')
    def test_actual_packet(self):self.adapter.validate(self.p)
    def test_nonfinite_time(self):
        for value in [float('inf'),float('nan')]:
            p=copy.deepcopy(self.p);p['timestamp_s']=value
            for c in p['cameras']:c['timestamp_s']=value
            with self.assertRaises(ValueError):self.adapter.validate(p)
    def test_gt_rejected(self):
        for key in ['ego_pose','actors','depth','split','visibility','future']:
            p=copy.deepcopy(self.p);p[key]=None
            with self.assertRaises(ValueError):self.adapter.validate(p)
    def test_frame_timestamp_rejected(self):
        for key in ['frame_id','timestamp_s']:
            p=copy.deepcopy(self.p);p['cameras'][2][key]+=1
            with self.assertRaises(ValueError):self.adapter.validate(p)
    def test_duplicate_and_swap(self):
        p=copy.deepcopy(self.p);p['cameras'][1]=p['cameras'][0]
        with self.assertRaises(ValueError):self.adapter.validate(p)
        p=copy.deepcopy(self.p);p['cameras'][0]['T_cam_to_ego']=p['cameras'][2]['T_cam_to_ego']
        with self.assertRaises(ValueError):self.adapter.validate(p)
    def test_gt_path_rejected(self):
        p=copy.deepcopy(self.p);p['cameras'][0]['path']='/tmp/gt.png'
        with self.assertRaises(ValueError):self.adapter.validate(p)
    def test_center_dimensions_velocity(self):
        b=old_boxes_to_hs([[10,3,1,2,5,1.5,-np.pi/2,2,-3]])[0]
        np.testing.assert_allclose(b,[10,3,1.75,5,2,1.5,0,2,-3])
        corners=corners_hs(b);np.testing.assert_allclose(corners.max(0)-corners.min(0),[5,2,1.5])
    def test_yaw_quarter_turn(self):
        b=old_boxes_to_hs([[10,3,1,2,5,1.5,0,0,0]])[0]
        np.testing.assert_allclose(corners_hs(b).max(0)-corners_hs(b).min(0),[2,5,1.5])
    def test_pitch_projection_and_negative_depth(self):
        for pitch in [0,.15,-.2]:
            c,s=np.cos(pitch),np.sin(pitch);r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            for camera in self.p['cameras']:
                T=np.array(camera['T_cam_to_ego']);T[:3,:3]=r@T[:3,:3];K=A@camera['K'];P=np.eye(4);P[:3,:3]=K;P=P@np.linalg.inv(T)
                x=np.array([1,2,10,1]);ego=T@x;uv=P@ego
                np.testing.assert_allclose(uv[:3],K@x[:3],atol=1e-10)
                x[2]=-10;self.assertLess((P@T@x)[2],0)
    def test_all_frozen_packets(self):
        root=Path('/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot/rgb_bundle')
        for f in sorted(root.glob('s*/packet.json')):self.adapter.validate(json.loads(f.read_text()))
if __name__=='__main__':unittest.main()
