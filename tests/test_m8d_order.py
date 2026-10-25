import copy
import unittest
import numpy as np
from datasets.m8d_camera_order import permute, check_pair, PERMUTATION, INVERSE
from datasets.m8b_petr_adapter import ORDER


class CameraPermutation(unittest.TestCase):
    def fixture(self):
        a = np.arange(72).reshape(1,6,3,2,2)
        k = [np.diag([float(i+2), float(i+3),1]) for i in range(6)]
        t = [np.eye(4) for _ in range(6)]
        for i in range(6):
            t[i][0,3] = i
        intr = [np.eye(4) for _ in range(6)]
        for i in range(6):
            intr[i][:3,:3] = k[i]
        ext = [np.linalg.inv(v).T for v in t]
        return a, dict(camera_names=ORDER, K=k, T_cam_to_ego=t, intrinsics=intr, extrinsics=ext,
              lidar2img=[i@e.T for i,e in zip(intr,ext)], rgb_sha256=list('abcdef'),
              frame_id=42,timestamp_s=2.1,token='synthetic',img_shape=[(2,2,3)]*6)

    def test_joint_permutation_projection(self):
        a,m = self.fixture(); b,n = permute(a,m,PERMUTATION)
        self.assertEqual(check_pair(a,m,b,n)['status'],'PermutationContractPassed')

    def test_roundtrip_and_no_mutation(self):
        a,m = self.fixture(); b,n=permute(a,m,PERMUTATION); c,o=permute(b,n,INVERSE)
        np.testing.assert_array_equal(a,c); self.assertEqual(o['camera_names'],ORDER)
        n['K'][0][0,0]=99; self.assertEqual(m['K'][0][0,0],2)

    def test_image_only_reorder_rejected(self):
        a,m=self.fixture(); b,n=permute(a,m,PERMUTATION); n['lidar2img']=m['lidar2img']
        with self.assertRaises(AssertionError): check_pair(a,m,b,n)

    def test_calibration_only_reorder_rejected(self):
        a,m=self.fixture(); b,n=permute(a,m,PERMUTATION)
        with self.assertRaises(AssertionError): check_pair(a,m,a,n)

    def test_timestamp_rejected(self):
        a,m=self.fixture(); b,n=permute(a,m,PERMUTATION); n['timestamp_s']+=.05
        with self.assertRaises(AssertionError): check_pair(a,m,b,n)

    def test_unknown_camera_field_rejected(self):
        a,m=self.fixture(); m['new_camera_field']=[1]*6
        with self.assertRaises(ValueError): permute(a,m,PERMUTATION)

    def test_non_permutation_rejected(self):
        a,m=self.fixture()
        with self.assertRaises(ValueError): permute(a,m,[0]*6)

if __name__=='__main__': unittest.main()
