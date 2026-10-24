import unittest
import numpy as np
from diagnostics.m8b_metrics import natural,conditional,project_bounds,upright_iou,iou2
from datasets.m8b_petr_adapter import corners_hs
class Metrics(unittest.TestCase):
    def test_serialized_bounds(self):self.assertAlmostEqual(iou2([0,0,4,2],np.array([0,0,4,2])),1)
    def test_duplicates_and_unknown(self):
        g=[dict(type='vehicle',box=[10,0,1,4,2,2,0,0,0],crop_retained=True)]
        p=dict(boxes=[[10,0,1,4,2,2,0,0,0],[10.5,0,1,4,2,2,0,0,0],[30,0,1,4,2,2,0,0,0]],scores=[.8,.7,.6],labels=[0,0,0])
        r=natural(p,g);self.assertEqual([r[k] for k in ['TP','FP','FN','duplicates']],[1,2,0,1])
    def test_wrong_class_not_match(self):
        p=dict(boxes=[[10,0,1,4,2,2,0,0,0]],scores=[.8],labels=[8]);g=[dict(type='vehicle',box=p['boxes'][0],crop_retained=True)]
        self.assertEqual(natural(p,g)['TP'],0)
    def test_conditional_does_not_cut_center_error(self):
        p=dict(boxes=[[100,0,1,4,2,2,0,0,0]],scores=[.8]);g=[dict(box=[10,0,1,4,2,2,0,0,0])]
        self.assertEqual(conditional(p,g,np.array([[.3]])),[[0,0]])
    def test_global_cardinality(self):
        p=dict(scores=[.8,.7]);matrix=np.array([[.5,.3],[.2,0.]])
        self.assertEqual(len(conditional(p,[{},{}],matrix)),2)
    def test_empty(self):self.assertEqual(conditional(dict(scores=[]),[],np.empty((0,0))),[])
    def test_iou(self):
        b=np.array([10,0,1,4,2,2,.3,0,0]);self.assertAlmostEqual(upright_iou(b,b),1)
        c=b.copy();c[0]+=100;self.assertEqual(upright_iou(b,c),0)
    def test_near_plane_clipping(self):
        K=np.array([[300,0,320],[0,300,192],[0,0,1.]])
        self.assertIsNone(project_bounds(corners_hs([0,0,-10,2,2,2,0]),K,np.eye(4)))
        self.assertIsNotNone(project_bounds(corners_hs([0,0,1,2,2,4,0]),K,np.eye(4)))
if __name__=='__main__':unittest.main()
