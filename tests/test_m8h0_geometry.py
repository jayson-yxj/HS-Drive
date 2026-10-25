import unittest
import numpy as np
import yaml
from pathlib import Path
from datasets.m8h0_geometry import make_box,footprint,pair_relation,visibility
from tools.m8h0_templates import segment_box_interval
A=dict(extent=[2,1,1],local_box_matrix=np.eye(4).tolist())
def box(x=0,y=0,z=1):
 m=np.eye(4);m[:3,3]=[x,y,z];return make_box(m,A)
class Geometry(unittest.TestCase):
 def test_full_dimensions(self):self.assertEqual(box()['dimensions_lwh_m'],[4.,2.,2.])
 def test_not_center_distance(self):self.assertAlmostEqual(pair_relation(box(),box(5))['xy_polygon_distance_m'],1)
 def test_xy_overlap(self):self.assertTrue(pair_relation(box(),box(3))['conservative_3d_overlap'])
 def test_z_separation_does_not_approve_horizontal_layout(self):
  r=pair_relation(box(),box(z=5));self.assertFalse(r['conservative_3d_overlap']);self.assertFalse(r['pair_clearance_pass'])
 def test_contact_not_clearance(self):self.assertFalse(pair_relation(box(),box(4))['pair_clearance_pass'])
 def test_conservative_line_warning(self):self.assertIsNotNone(segment_box_interval([-10,0,1],[10,0,1],box()))
 def test_parallel_miss(self):self.assertIsNone(segment_box_interval([-10,3,1],[10,3,1],box()))
 def test_segment_behind(self):self.assertIsNone(segment_box_interval([-10,0,1],[-5,0,1],box()))
 def test_camera_front_and_rear(self):
  rig=yaml.safe_load(Path('configs/cameras.yaml').read_text())
  for x,name in [(30,'front'),(-30,'rear')]:
   r=visibility(np.eye(4),box(x),rig);self.assertTrue(r['in_20_40m']);self.assertTrue(next(v['center_in_frustum'] for v in r['views'] if v['camera']==name));self.assertFalse(r['visibility_certified'])
 def test_distance_is_not_visibility(self):
  rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());r=visibility(np.eye(4),box(30,z=100),rig);self.assertTrue(r['in_20_40m']);self.assertFalse(r['frustum_supported'])
 def test_rotation_equivariance(self):
  rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());m=np.eye(4);m[:2,:2]=[[0,-1],[1,0]];r=visibility(m,box(y=30),rig);self.assertTrue(next(v['center_in_frustum'] for v in r['views'] if v['camera']=='front'))
 def test_native_height_not_rewritten(self):
  m=np.eye(4);m[2,3]=.5;before=m.copy();b=make_box(m,A);np.testing.assert_equal(m,before);self.assertEqual(b['center_world_m'][2],.5)
if __name__=='__main__':unittest.main()
