import unittest
import numpy as np
from datasets.m8g_geometry import OpenDriveGeometry,inside_box,overhead_class,planned_box
from tools.m8g_history import preflight

XODR='''<OpenDRIVE><road id="1" length="100"><planView><geometry s="0" x="0" y="0" hdg="0" length="100"><line/></geometry></planView><lanes><laneSection s="0"><left><lane id="1" type="driving"><width sOffset="0" a="4" b="0" c="0" d="0"/></lane></left><right><lane id="-1" type="driving"><width sOffset="0" a="3" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>'''
def box(z=1,e=(1,1,1)):
 m=np.eye(4);m[2,3]=z;cs=np.array([[x,y,zz] for x in [-e[0],e[0]] for y in [-e[1],e[1]] for zz in [-e[2],e[2]]])+m[:3,3]
 return dict(T_world_from_box=m.tolist(),extent_m=list(e),corners_world_m=cs.tolist(),bottom_z=float(z-e[2]),top_z=float(z+e[2]))
class Geometry(unittest.TestCase):
 def test_positive_lane_carla_y_flip(self):np.testing.assert_allclose(OpenDriveGeometry(XODR).lane_center(1,1,20),[20,-2,0])
 def test_negative_lane(self):np.testing.assert_allclose(OpenDriveGeometry(XODR).lane_center(1,-1,20),[20,1.5,0])
 def test_frozen_margin(self):
  g=OpenDriveGeometry(XODR);self.assertTrue(g.point(1,1,20,[20,-3.85,0])['within']);self.assertFalse(g.point(1,1,20,[20,-3.95,0])['within'])
 def test_wrong_elevation_refused(self):self.assertFalse(OpenDriveGeometry(XODR).point(1,1,20,[20,-2,2])['within'])
 def test_beyond_road_endpoint_refused(self):self.assertFalse(OpenDriveGeometry(XODR).point(1,1,99,[102,-2,0])['within'])
 def test_arc_center(self):
  xml=XODR.replace('<line/>','<arc curvature="0.01"/>');g=OpenDriveGeometry(xml);s=20.;ref=np.array([np.sin(.2)/.01,(1-np.cos(.2))/.01]);want=ref+np.array([-np.sin(.2),np.cos(.2)])*2
  np.testing.assert_allclose(g.lane_center(1,1,s)[:2],[want[0],-want[1]],atol=1e-8)
 def test_unsupported_refuses(self):
  with self.assertRaises(ValueError):OpenDriveGeometry(XODR.replace('<line/>','<spiral curvStart="0" curvEnd="1"/>')).reference(1,20)
 def test_box_rotation(self):
  b=box();m=np.array(b['T_world_from_box']);m[:2,:2]=[[0,-1],[1,0]];b['T_world_from_box']=m.tolist();self.assertTrue(inside_box([.8,.8,1],b,0));self.assertFalse(inside_box([2,0,1],b,0))
 def test_high_hit_without_source_unknown(self):self.assertEqual(overhead_class({'xyz':[0,0,10]},box(),[],True),'UnidentifiedRayHit')
 def test_high_hit_not_full_source_clearance(self):self.assertEqual(overhead_class({'xyz':[0,0,10]},box(),[box(5,(1,1,5))],True),'UnidentifiedRayHit')
 def test_independent_source_clearance(self):self.assertEqual(overhead_class({'xyz':[0,0,5]},box(),[box(5)],True),'VerifiedNonIntersectingOverhead')
 def test_render_required(self):self.assertEqual(overhead_class({'xyz':[0,0,5]},box(),[box(5)],False),'UnidentifiedRayHit')
 def test_body_intrusion_not_overhead(self):self.assertEqual(overhead_class({'xyz':[0,0,1]},box(),[box()],True),'SourceQualifiedCollision')
 def test_competing_sources(self):self.assertEqual(overhead_class({'xyz':[0,0,1]},box(),[box(),box()],True),'ConflictingGeometry')
 def test_no_surface_no_box(self):
  r=dict(asset=dict(extent=[1,1,1],local_box_matrix=np.eye(4).tolist()),waypoint=dict(matrix=np.eye(4).tolist()),rays=[dict(start=[0,0,8],decision=dict(selected=None))]);self.assertIsNone(planned_box(r))
 def test_history_footprint_not_center(self):
  p=[[-1,-1],[1,-1],[1,1],[-1,1]];r=dict(map='Town02',target=True,center_xy=[4,0],radius_m=2,polygon=[[2,-1],[6,-1],[6,1],[2,1]],source='fixture',kind='ActorOBB',index_origin='synthetic')
  out=preflight(p,'Town02',dict(records=[r]));self.assertEqual(out['actor']['distance_m'],1);self.assertFalse(out['FinalEligible'])
 def test_empty_history_not_fresh(self):self.assertEqual(preflight([[0,0],[1,0],[1,1],[0,1]],'Town02',dict(records=[]))['status'],'HistoricalCoverageIncomplete')
if __name__=='__main__':unittest.main()
