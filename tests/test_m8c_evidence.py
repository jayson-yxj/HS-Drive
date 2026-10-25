import ast,json,unittest
from pathlib import Path
import numpy as np
class Evidence(unittest.TestCase):
    def test_untrusted_identity_has_no_error(self):
        report=json.loads(Path('reports/m8c_2d_3d_association_audit.json').read_text())
        for row in report['rows']:
            self.assertEqual(row['metric_error'] is not None,row['grade']=='IdentitySupported')
        self.assertEqual(sum(r['grade']=='IdentitySupported' and r['range_m']>=20 for r in report['rows']),0)
    def test_no_physical_B(self):
        r=json.loads(Path('reports/m8c_frozen_ablation.json').read_text());self.assertEqual(r['B_runs'],0);self.assertFalse(r['improvement_claim'])
    def test_hook_did_not_change_outputs(self):
        r=json.loads(Path('reports/m8c_position_embedding_analysis.json').read_text());self.assertTrue(r['hook_on_off_bitwise_equal']);self.assertTrue(r['baseline_output_bitwise_equal']);self.assertFalse(r['computed_mask_consumed'])
    def test_inference_no_backward_or_pickle(self):
        tree=ast.parse(Path('tools/m8c_gpu_hook.py').read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('backward',attrs);self.assertNotIn('step',attrs)
        source=Path('tools/m8c_gpu_hook.py').read_text();self.assertIn('allow_pickle=False',source);self.assertIn('strict=True',source)
    def test_semantics_authority(self):
        e=json.loads(Path('reports/m8c_semantic_schema.json').read_text())['enum']
        self.assertEqual([e[k] for k in ['Walls','Fences','GuardRail']],[4,5,28])
    def test_known_gt_not_frame_independent(self):
        r=json.loads(Path('reports/m8c_2d_3d_association_audit.json').read_text())['rows']
        self.assertEqual(len(r),155);self.assertLess(len({v['group'] for v in r}),155)
if __name__=='__main__':unittest.main()
