import unittest
import numpy as np
from diagnostics.m8c_geometry import camera_in_reference,upright_boxes_to_ego,association_grade,pixel_support
from datasets.m8b_petr_adapter import corners_hs
class Geometry(unittest.TestCase):
    def test_projection_yaw_translation(self):
        for yaw in [0,.37,-1.8]:
            c,s=np.cos(yaw),np.sin(yaw);T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[2,-1,1.3]
            E_C=np.eye(4);E_C[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];E_C[:3,3]=[1,0,1.5]
            R_C=camera_in_reference(T,E_C);points=np.array([[5,2,1,1],[20,-5,3,1]]).T;Rpoints=np.linalg.inv(T)@points
            np.testing.assert_allclose(np.linalg.inv(R_C)@Rpoints,np.linalg.inv(E_C)@points,atol=1e-12)
    def test_box_corners_velocity(self):
        T=np.eye(4);a=.4;T[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];T[:3,3]=[1,2,3]
        b=np.array([[10,-3,1,5,2,1.6,-.2,4,-2.]])
        result=upright_boxes_to_ego(b,T)[0];expected=corners_hs(b[0])@T[:3,:3].T+T[:3,3]
        np.testing.assert_allclose(corners_hs(result),expected,atol=1e-12)
        np.testing.assert_allclose(result[7:9],(T[:3,:3]@[4,-2,0])[:2])
        np.testing.assert_allclose(upright_boxes_to_ego([result],np.linalg.inv(T)),b,atol=1e-12)
    def test_tilt_no_silent_yaw(self):
        T=np.eye(4);a=.1;T[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
        with self.assertRaises(ValueError):upright_boxes_to_ego([[1,2,3,5,2,1,0,0,0]],T)
    def test_reflection_rejected(self):
        T=np.eye(4);T[1,1]=-1
        with self.assertRaises(ValueError):camera_in_reference(T,np.eye(4))
    def test_identity_no_error_dependency(self):
        good=dict(retained_pixels=20,GT_association_purity=.99,instance_bbox_iou=.6,mask_recall=.9,competing_bbox_iou=0)
        self.assertEqual(association_grade([good,good],True),'IdentitySupported')
        self.assertEqual(association_grade([good],True),'ProjectionProxyOnly')
        self.assertEqual(association_grade([good,dict(good,competing_bbox_iou=.2)],True),'AmbiguousAssociation')
        self.assertEqual(association_grade([],False),'NoSupportedAssociation')
    def test_low_iou_not_identity(self):
        v=dict(retained_pixels=20,GT_association_purity=.99,instance_bbox_iou=.11,mask_recall=.9,competing_bbox_iou=0)
        self.assertEqual(association_grade([v,v],True),'ProjectionProxyOnly')
    def test_pixel_support(self):
        m=np.zeros((10,10),bool);m[2:4,3:7]=1;r=pixel_support(m,[3,2,7,4]);self.assertEqual(r['pixels'],8);self.assertEqual(r['mask_recall'],1);self.assertAlmostEqual(r['bbox_iou'],1)
if __name__=='__main__':unittest.main()
