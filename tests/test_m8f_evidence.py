import json,unittest
from pathlib import Path
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8f-grounded-quality')
def report(name):return json.loads(Path('reports/m8f_'+name+'.json').read_text())
class ActualEvidence(unittest.TestCase):
 def test_smoke_stops_expansion(self):
  r=report('natural_scene_quality');self.assertEqual(r['overall'],'SmokeNoGo');self.assertEqual(r['sensor_records'],192);self.assertTrue(all(x['ok'] for x in r['sensor_sync_and_sha']))
 def test_every_request_kept(self):
  r=report('object_eligibility');self.assertEqual(len(r['objects']),8);self.assertEqual(sum(x['accepted_for_spawn'] for x in r['objects']),1);self.assertEqual(r['eligible'],0)
 def test_all_ground_rays_have_sources_no_fallback(self):
  r=report('ray_hit_and_contact_audit');self.assertEqual(r['preflight_rays'],240)
  for q in r['requests']:
   for x in q['preflight'].values():
    self.assertTrue(x['no_fallback']);self.assertEqual(len(x['rays']),15)
    for ray in x['rays']:self.assertEqual(set(['start','end','hits','frame','decision'])-set(ray),set())
 def test_final_actor_not_creation_only(self):
  for phase in ['present','shifted']:
   d=json.loads((ROOT/'capture/m8f_development_01/evaluator'/(phase+'_support.json')).read_text())
   c=d['checks'][0]['final'];self.assertEqual(c['status'],'GroundedByGeometryProxy');self.assertFalse(c['PhysicalContactCertified']);self.assertEqual(len(c['rays']),15);self.assertLess(max(c['support_gaps_m']),.12)
 def test_rejected_controls_and_no_false_whole_scene_precision(self):
  r=report('negative_controls');self.assertTrue(all(x['passed'] for x in r['rows'] if 'passed' in x));self.assertTrue(report('identity_and_background_audit')['no_physical_precision_claim'])
 def test_no_false_final_or_training(self):
  self.assertEqual(report('independence_exceptions')['new_final_groups'],0);self.assertEqual(report('training_readiness')['status'],'TrainingReadinessNoGo');self.assertEqual(report('far_field_support')['independent_split_eligible_far'],0)
 def test_counterfactual_identity_not_repeat_count(self):
  r=report('counterfactual_regression');self.assertEqual(r['qualifying_present_objects'],1);self.assertTrue(r['all_pass']);self.assertEqual(report('far_field_support')['natural_geometry_render_far'],1)
if __name__=='__main__':unittest.main()
