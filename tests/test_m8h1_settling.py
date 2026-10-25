import copy,json,unittest
from pathlib import Path
from datasets.m8h1_settling import stationary,drift
C=json.loads(Path('configs/m8h1/protocol.json').read_text())['settling']
def row(x=0,z=0,v=0,yaw=0):return dict(pose=dict(xyz=[x,0,z],rotation=[0,yaw,0]),velocity_mps=[v,0,0])
class Settling(unittest.TestCase):
 def test_window_required(self):self.assertFalse(stationary([row()]*20,C)['stable'])
 def test_stationary(self):self.assertTrue(stationary([row()]*40,C)['stable'])
 def test_raw_velocity_not_smoothed(self):
  r=[row() for _ in range(40)];r[-3]=row(v=.1);self.assertFalse(stationary(r,C)['stable'])
 def test_moving_position(self):self.assertFalse(stationary([row(x=i*.01) for i in range(40)],C)['stable'])
 def test_yaw_wrap(self):self.assertTrue(stationary([row(yaw=359.99 if i%2 else .01) for i in range(40)],C)['stable'])
 def test_vertical_settling_reported(self):
  d=drift(row(z=.5)['pose'],row(z=.22)['pose'],C);self.assertTrue(d['passed']);self.assertAlmostEqual(d['z_m'],-.28)
 def test_original_xy_limit_preserved(self):self.assertFalse(drift(row()['pose'],row(x=.021)['pose'],C)['passed'])
if __name__=='__main__':unittest.main()
