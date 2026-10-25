"""Actual-evidence invariants; synthetic geometry tests are separate."""
import json
import unittest
from pathlib import Path
from tools.m8g_common import ROOT,sha

def report(n):return json.loads(Path('reports/m8g_'+n+'.json').read_text())
class Evidence(unittest.TestCase):
 def test_full_differential_not_status_only(self):
  r=report('validator_differential');self.assertEqual(r['phase_count'],16);self.assertTrue(all(n==16 for n in r['frozen_reproduction'].values()));self.assertEqual(r['newly_accepted'],[])
 def test_all_hits_and_bottom_corners(self):
  rs=report('lane_boundary_differential')['records'];self.assertEqual(len(rs),240)
  self.assertEqual(sum(r['is_bottom_corner'] for r in rs),64)
  self.assertTrue(all('ground_hits' in r and 'original_point' in r for r in rs))
 def test_no_unproven_clearance_override(self):
  for name in ['none_hit_analysis','overhead_clearance']:
   r=report(name);self.assertFalse(r['acceptance_override'])
   for x in r['records']:
    if x['status']=='VerifiedNonIntersectingOverhead':self.assertTrue(len(x['plausible_sources'])==1 and any(e['agrees'] for e in x['render_evidence']))
 def test_trigger_evidence_is_hypothesis(self):
  r=report('source_geometry_followup');self.assertEqual(r['trigger_supported_hits'],21)
  for row in r['ray_rows']:self.assertFalse(row['placement_override'])
 def test_history_original_and_raw_sources(self):
  r=report('historical_preflight_index');p=Path(r['index']['path']);self.assertEqual(sha(p),r['index']['sha256']);d=json.loads(p.read_text())
  self.assertEqual(len(d['records']),16562);self.assertEqual({x['index_origin'] for x in d['records']},{'historical_index.json','historical_raw_frame_supplement.json'})
  self.assertTrue(all(x['source_sha256'] for x in d['records']))
 def test_near_neighbor_regression(self):
  rows=report('historical_preflight_index')['regression'];dist=[r['actual']['actor']['distance_m'] for r in rows]
  for v in [3.6356264669372553,4.595351322041442]:self.assertTrue(any(abs(d-v)<1e-6 for d in dist))
 def test_every_candidate_preflight_before_selection(self):
  d=json.loads((ROOT/'scan_all_candidates.json').read_text());self.assertEqual(len(d['records']),80)
  for r in d['records']:
   self.assertTrue(r['history_evaluated_before_acceptance'])
   self.assertEqual(set(r['historical_preflight']),{'present','shifted'})
   for p in r['historical_preflight'].values():self.assertFalse(p['preflight']['FinalEligible'])
 def test_no_cherry_picked_scan_successes(self):
  scan=report('feasibility_scan');r=json.loads(Path(scan['raw']['path']).read_text())['records'];self.assertEqual(sum(x['geometry_pass'] for x in r),28);self.assertEqual(sum(not x['geometry_pass'] for x in r),52)
  self.assertFalse(scan['pre_smoke_core_geometry_pass'])
 def test_class_scene_semantics_and_thresholds(self):
  a=json.loads(Path('configs/m8f/contract.json').read_text());b=json.loads(Path('configs/m8g/capture_contract.json').read_text());self.assertEqual(a['ground'],b['ground'])
  for k in a['class_regions']:
   for f in ['lanes','semantics']:self.assertEqual(a['class_regions'][k][f],b['class_regions'][k][f])
 def test_illegal_controls_still_rejected(self):
  r=report('positive_negative_controls');self.assertTrue(r['all_passed']);self.assertEqual(len(r['records']),10)
  for x in r['records']:self.assertEqual(x['A'],x['B'])
 def test_failed_precondition_does_not_capture(self):
  r=report('development_smoke');self.assertEqual(r['status'],'NotRun');self.assertEqual(r['new_RGB_frames'],0);self.assertFalse((ROOT/'capture').exists());self.assertEqual(r['attempts'],0)
 def test_training_and_final_still_no_go(self):
  r=report('training_readiness');self.assertEqual(r['status'],'TrainingReadinessNoGo');self.assertEqual(r['new_Final'],0);self.assertFalse(r['training_authorized']);self.assertFalse(r['visual_control_authorized'])
 def test_source_revision_available(self):
  r=report('validator_differential')
  for path,digest in r['sources'].items():self.assertTrue(sha(path)==digest or (ROOT/'source_revisions'/digest/path).is_file(),path)
if __name__=='__main__':unittest.main()
