import unittest
from datasets.m8f_quality import choose_surface,polygon,polygon_distance,eligibility
C=dict(layer_window_m=8,distinct_layer_m=.5,waypoint_z_tolerance_m=.5)
def h(z,label=1):return dict(xyz=[0,0,z],semantic=label)
class Quality(unittest.TestCase):
 def test_no_hit_unknown(self):self.assertEqual(choose_surface([],0,[1],C)['status'],'GroundSupportUnknown')
 def test_water_rejected(self):self.assertIn('WaterSurface',choose_surface([h(0,23)],0,[1],C)['reasons'])
 def test_multilevel_rejected(self):self.assertIn('MultipleGroundLayers',choose_surface([h(0),h(4)],0,[1],C)['reasons'])
 def test_wrong_layer_rejected(self):self.assertIsNone(choose_surface([h(-3)],0,[1],C)['selected'])
 def test_roof_rejected(self):self.assertIn('CoveredOrWrongTopSurface',choose_surface([h(0),h(2,3)],0,[1],C)['reasons'])
 def test_valid_proxy(self):self.assertEqual(choose_surface([h(.2)],0,[1],C)['status'],'GroundedByGeometryProxy')
 def test_footprint_distance_not_centers(self):self.assertEqual(polygon_distance(polygon([0,0],[3,1],0),polygon([5,0],[3,1],0)),0)
 def test_rotated_distance(self):self.assertAlmostEqual(polygon_distance(polygon([0,0],[1,1],0),polygon([4,0],[1,1],0)),2)
 def test_all_gates_conjunctive(self):
  good=dict(RenderIdentityGrade='RenderLinkedSingleView',GroundSurfaceGrade='GroundedByGeometryProxy',NaturalPlacementGrade='NaturalByGeometryProxy',AnnotationGrade='SourceContractComplete',SplitIndependenceGrade='SpatiallySeparated',InputSourceGrade='CompleteSynchronized')
  self.assertEqual(eligibility(good)['status'],'EligibleForProposalOnly')
  for k in good:
   bad=dict(good);bad[k]='Unknown';self.assertEqual(eligibility(bad)['status'],'NotEligible')
 def test_no_control_authorization(self):self.assertFalse(eligibility({})['training_authorized'])
if __name__=='__main__':unittest.main()
