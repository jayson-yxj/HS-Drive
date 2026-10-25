import unittest
from diagnostics.m8d_evaluation import paired_counts,localization
from diagnostics.m8c_geometry import association_grade
from tools.m8d_common import stats

class PairedEvaluation(unittest.TestCase):
    def test_all_four_transitions(self):
        rows=[dict(A=a,B=b) for a in [True,False] for b in [True,False]]
        r=paired_counts(rows)
        self.assertEqual(r['GT'],4);self.assertEqual(r['delta_TP'],0)
        self.assertEqual(list(r['transitions'].values()),[1,1,1,1])
    def test_empty_not_perfect(self):
        self.assertIsNone(paired_counts([])['A_recall'])
        self.assertIsNone(localization([],stats)['over5_rate'])
    def test_untrusted_has_no_metric(self):
        r=localization([dict(group='g',metric_error=None)],stats)
        self.assertEqual(r['n'],0);self.assertIsNone(r['statistics']['XY_m']['p95'])
    def test_far_no_identity_upgrade(self):
        v=dict(retained_pixels=30,GT_association_purity=1,instance_bbox_iou=.2,mask_recall=.95,competing_bbox_iou=0)
        self.assertEqual(association_grade([v,v],True),'ProjectionProxyOnly')
if __name__=='__main__':unittest.main()
