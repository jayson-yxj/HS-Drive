import unittest
import numpy as np
from datasets.m8e_reference import taxonomy,points_inside,render_grade,prediction_grade

class GroundTruthContract(unittest.TestCase):
    def test_static_semantic_not_automatically_barrier(self):
        self.assertEqual(taxonomy([4],'wall')[0],'Unknown')
        self.assertEqual(taxonomy([21],'unknown_prop')[0],'Unknown')
        self.assertEqual(taxonomy([],'static.prop.streetbarrier')[0],'barrier')
        self.assertEqual(taxonomy([],'static.prop.trafficcone01')[0],'traffic_cone')
    def test_single_view_is_valid_evidence(self):
        self.assertEqual(render_grade([dict(accepted=True)]),'RenderLinkedSingleView')
        self.assertEqual(render_grade([dict(accepted=True),dict(accepted=True)]),'RenderLinkedMultiView')
    def test_conflicting_sources_rejected(self):
        self.assertEqual(render_grade([dict(accepted=True,identity_conflict=True)]),'AmbiguousIdentity')
    def test_rotated_bbox_not_axis_aligned(self):
        t=np.eye(4);t[:2,:2]=[[0,-1],[1,0]];t[:3,3]=[10,0,0]
        np.testing.assert_array_equal(points_inside([[10,1.5,0],[11.5,0,0]],t,[2,.5,1],0),[True,False])
    def test_predict_identity_no_3d_error_gate(self):
        v=dict(source_linked=True,retained_pixels=20,instance_bbox_iou=.6,mask_recall=.9)
        self.assertEqual(prediction_grade([v],True),'RenderLinkedSingleView')
        self.assertEqual(prediction_grade([dict(v,source_linked=False)],True),'ProjectionProxyOnly')
    def test_competing_object_remains_ambiguous(self):
        v=dict(source_linked=True,retained_pixels=20,instance_bbox_iou=.6,mask_recall=.9,competing_bbox_iou=.2)
        self.assertEqual(prediction_grade([v],True),'AmbiguousIdentity')
    def test_duplicate_component_not_multiview(self):
        v=dict(camera='front',accepted=True)
        self.assertEqual(render_grade([v,v]),'RenderLinkedSingleView')
    def test_unknown_not_free(self):
        self.assertEqual(render_grade([]),'UnlabeledOrUnknown')
        self.assertEqual(prediction_grade([],False),'UnlabeledOrUnknown')
if __name__=='__main__':unittest.main()
