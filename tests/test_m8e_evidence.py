"""Checks of actual data source separation and preserved uncertainty."""
import json,unittest
from pathlib import Path
from datasets.m8e_reference import render_grade
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8e-groundtruth')
def report(n):return json.loads(Path('reports/m8e_'+n+'.json').read_text())
class Evidence(unittest.TestCase):
    def test_exact_bounded_capture_and_sync(self):
        r=report('actor_environment_gt')
        self.assertEqual(r['frames'],32);self.assertEqual(r['groups'],8)
        self.assertEqual(r['sensor_records'],768);self.assertTrue(r['all24sensor_payloads_verified'])
    def test_no_GT_in_input_packet(self):
        for a in report('actor_environment_gt')['RGB_only_packets']:
            p=json.loads(Path(a['path']).read_text())
            self.assertEqual(set(p),{'schema','token','camera_order','frame_id','timestamp_s','image_size','cameras'})
            self.assertEqual(len(p['cameras']),6)
            for c in p['cameras']:
                self.assertEqual(set(c),{'name','path','sha256','frame_id','timestamp_s','K','T_cam_to_ego'})
                self.assertIn('/rgb/',c['path']);self.assertNotIn('/evaluator/',c['path'])
    def test_untrusted_historical_errors_null(self):
        for r in report('historical_evaluator')['rows']:
            self.assertEqual(r['metric_error'] is not None,r['grade'].startswith('RenderLinked'))
    def test_distinct_camera_recount_unchanged(self):
        for a in report('actor_environment_gt')['audit_assets']:
            for f in json.loads(Path(a['path']).read_text())['frames']:
                for r in f['objects']:
                    self.assertEqual(render_grade(r['views']),r['identity_grade'])
    def test_no_fake_background_false_positive(self):
        r=report('background_annotation_audit')
        self.assertEqual(r['candidate_count'],154)
        self.assertTrue(all(not x['physical_false_positive_confirmed'] for x in r['records']))
    def test_heldout_exception_blocks_training(self):
        self.assertFalse(report('heldout_independence_exception')['usable_as_blind_final'])
        self.assertEqual(report('training_readiness')['status'],'TrainingReadinessNoGo')
        self.assertEqual(report('frozen_petr_diagnostic')['held_out_final_inference_calls'],0)
    def test_far_not_multiplied_by_views_or_phases(self):
        r=report('far_field_support')
        self.assertEqual(r['unique_present_far_targets'],len({(x['group'],x['entity_key']) for x in r['qualifying_objects']}))
        self.assertEqual(r['status'],'FarFieldSupportInsufficient')
if __name__=='__main__':unittest.main()
