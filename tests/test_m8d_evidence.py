"""Read-only checks against executed evidence and supervision boundaries."""
import ast
import json
import unittest
from pathlib import Path

def read(name):
    return json.loads(Path('reports/m8d_'+name+'.json').read_text())

class ExecutedEvidence(unittest.TestCase):
    def test_geometry_contract_all_physical_cameras(self):
        r=read('permutation_contract')
        self.assertEqual(r['frames'],192)
        self.assertTrue(r['inverse_exact'] and r['image_SHA_multiset_equal'])
        self.assertTrue(r['all_projection_residual_zero'])
        for frame in r['all_camera_identity_records']:
            self.assertEqual(len({x['name'] for x in frame['cameras']}),6)

    def test_preserved_GT_and_no_silent_dropped_positions(self):
        r=read('paired_detection_quality')
        self.assertEqual(len(r['all38_groups']),38)
        self.assertEqual(len(r['targets']),155)
        for v in ['A','B']:
            self.assertEqual(r['natural'][v]['TP']+r['natural'][v]['FN'],155)
        self.assertEqual(sum(r['strata']['all']['transitions'].values()),155)

    def test_identity_errors_cannot_leak_into_unsupported_rows(self):
        r=read('paired_identity_support')
        for v in ['A','B']:
            self.assertEqual(len(r['all_rows'][v]),155)
            for row in r['all_rows'][v]:
                self.assertEqual(row['metric_error'] is not None,row['grade']=='IdentitySupported')
        common=set(map(tuple,r['common_supported']))
        for v in ['A','B']:
            self.assertTrue(common.issubset({(x['token'],x['gt_index']) for x in r['all_rows'][v] if x['grade']=='IdentitySupported'}))

    def test_full_A_replay_and_weight_contract(self):
        r=read('latency_resource')
        self.assertEqual(len(r['full_A_replay']),192)
        self.assertTrue(all(x['box_exact'] and x['score_exact'] and x['labels_exact'] for x in r['full_A_replay']))

    def test_inference_module_no_evaluator_import_or_training(self):
        tree=ast.parse(Path('tools/m8d_gpu.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x and (x.startswith('diagnostics') or 'evaluate' in x) for x in imports))
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertNotIn('backward',attrs);self.assertNotIn('step',attrs)
        self.assertEqual(read('ablation_protocol')['protocol']['extra_B_orders_allowed'],0)

if __name__=='__main__':unittest.main()
