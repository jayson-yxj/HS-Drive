import ast,json,unittest
from pathlib import Path
import numpy as np
from tools.m8h1_common import ROOT,sha

def report(n):return json.loads(Path('reports/m8h1_'+n+'.json').read_text())
class Evidence(unittest.TestCase):
 def test_only_approved_templates(self):
  r=report('frozen_templates');self.assertEqual([t['id'] for t in r['templates']],['m8h0_template_00','m8h0_template_02']);self.assertFalse(r['FinalEligible'])
 def test_single_shift_source(self):
  r=report('shifted_contract');self.assertTrue(r['no_other_candidate_evaluated'])
  for g in r['groups']:
   self.assertEqual(g['status'],'ShiftedContractGo')
   for role in ['vehicle','person']:
    x=g['poses'][role]['phases'];self.assertEqual(set(x),{'present','shifted'});self.assertAlmostEqual(abs(x['shifted']['declared_waypoint']['s']-x['present']['declared_waypoint']['s']),3,places=4)
 def test_actual_settling_raw_window(self):
  for r in report('ego_settling')['rows']:
   self.assertEqual(r['settling_status'],'SettledPoseObserved');self.assertLessEqual(r['stationarity']['max_speed_mps'],.02);self.assertTrue(r['stationarity']['raw_no_filter']);self.assertLess(r['final_pose']['xyz'][2],.3)
 def test_no_hit_never_pass(self):
  for p in (ROOT/'capture').glob('*/ego_present_review.json'):
   d=json.loads(p.read_text());g=d['geometry'];self.assertEqual(g['status'],'GeometryNoGo');self.assertFalse(g['all15_support_observed']);self.assertEqual(len(g['observed_pose_support_result']['rays']),15);self.assertTrue(all(not r['hits'] for r in g['observed_pose_support_result']['rays']))
 def test_real_failure_stops_downstream(self):
  self.assertEqual(report('acceptance')['status'],'DevelopmentSmokeNoGo');self.assertEqual(report('acceptance')['targets_spawned'],0)
  self.assertFalse(list((ROOT/'capture').glob('*/evaluator/*')));self.assertFalse(list((ROOT/'capture').glob('*/rgb/*')))
 def test_invalid_startup_not_hidden(self):
  d=json.loads((ROOT/'invalid_measurement_attempt/reason.json').read_text());self.assertEqual(d['RGB_frames'],0)
  self.assertEqual(report('ego_settling')['all_spawn_calls_including_invalid'],4)
  for p in (ROOT/'invalid_measurement_attempt/capture').glob('*/ego_present_review.json'):self.assertEqual(json.loads(p.read_text())['initial_pose']['xyz'],[0.,0.,0.])
 def test_initial_reference_is_native_rpc(self):
  for p in (ROOT/'capture').glob('*/ego_present_review.json'):
   d=json.loads(p.read_text());self.assertEqual(d['initial_pose'],d['requested_spawn']);self.assertEqual(d['initial_pose_source'],'Frozen spawn RPC transform, not pre-tick cache')
 def test_no_disable_physics_autopilot_or_random_nav(self):
  forbidden={'set_simulate_physics','apply_physics_control','start_autopilot','set_autopilot','get_random_location_from_navigation'}
  for p in Path('tools').glob('m8h1_*.py'):
   tree=ast.parse(p.read_text());calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)};self.assertFalse(calls&forbidden,(p,calls&forbidden))
 def test_prior_ground_not_substituted(self):
  d=report('support_diagnostic')
  for r in d['rows']:
   self.assertFalse(r['acceptance_override'])
   self.assertTrue(all(p['ray_start_minus_prior_ground_m']<0 for p in r['point_rows']))
 def test_unknown_identity_not_detection_failure(self):
  for n in ['actual_visibility','render_identity','counterfactual_identity','sensor_synchronization']:
   r=report(n);self.assertEqual(r['status'],'NotRun');self.assertEqual(r['new_keyframes'],0);self.assertTrue(r['unobservable_target_not_counted_as_detection_failure'])
 def test_training_stays_closed(self):
  r=report('training_readiness');self.assertEqual(r['status'],'TrainingReadinessNoGo');self.assertFalse(r['FinalEligible']);self.assertEqual(r['new_training_samples'],0)
 def test_historical_thresholds_identical(self):
  frozen=report('frozen_templates')['original_geometry_contract'];self.assertEqual(sha(frozen['path']),frozen['sha256']);self.assertEqual(json.loads(Path(frozen['path']).read_text())['ground']['lane_inset_m'],.1)
if __name__=='__main__':unittest.main()
