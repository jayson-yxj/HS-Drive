import ast,json,unittest
from pathlib import Path
import numpy as np
from tools.m8h0_common import ROOT,HISTORY,sha

def report(name):return json.loads(Path('reports/m8h0_'+name+'.json').read_text())
class Evidence(unittest.TestCase):
 def test_native_inventory_complete(self):
  raw=json.loads((ROOT/'native_spawn_points_raw.json').read_text());full=json.loads((ROOT/'vehicle_inventory_full.json').read_text());self.assertEqual(len(raw['records']),101);self.assertEqual(len(full['records']),101)
  for a,b in zip(raw['records'],full['records']):self.assertEqual(a['transform'],b['transform'])
 def test_navigation_calls_and_actual_positions(self):
  raw=[json.loads(s) for s in (ROOT/'navigation_returns.jsonl').read_text().splitlines()];full=json.loads((ROOT/'navigation_inventory_full.json').read_text());self.assertEqual(len(raw),60)
  self.assertEqual([r['request_order'] for r in raw],list(range(60)))
  for a,b in zip(raw,full['records']):self.assertEqual(a['xyz'],b['transform']['xyz'])
 def test_candidate_does_not_equal_settled_actor(self):
  d=json.loads((ROOT/'vehicle_inventory_full.json').read_text())
  self.assertEqual(sum(r['geometry']['status']=='GeometryCandidateProxySupported' for r in d['records']),42)
  for r in d['records']:
   g=r['geometry'];self.assertTrue(g['not_a_spawn_authorization']);self.assertEqual(len(g['rays']),15);self.assertEqual(len(g['lane_checks']),15);self.assertAlmostEqual(g['native_z_offset_to_waypoint_m'],.5)
   m=np.array(r['transform']['matrix']);h=np.array(g['hypothetical_actor_matrix']);np.testing.assert_allclose(m[:2,:],h[:2,:],atol=0)
 def test_template_count_source_and_no_actor(self):
  d=report('curated_template_candidates');self.assertEqual(len(d['templates']),4);self.assertEqual(d['new_actors'],0);self.assertEqual(d['new_RGB'],0);self.assertTrue(d['proposal_not_execution_approval'])
  for t in d['templates']:
   for k in ['ego','vehicle','person']:self.assertEqual(t[k]['geometry']['status'],'GeometryCandidateProxySupported');self.assertTrue(t['historical_preflight'][t[k]['id']]['before_template_selection'])
   self.assertTrue(all(p['pair_clearance_pass'] for p in t['full_obb_pairs'].values()))
 def test_far_frustum_not_visibility(self):
  for t in report('curated_template_candidates')['templates']:
   for v in t['nominal_views'].values():self.assertTrue(v['in_20_40m']);self.assertTrue(v['frustum_supported']);self.assertFalse(v['visibility_certified'])
 def test_history_and_near_samples(self):
  r=report('historical_preflight');self.assertEqual(r['records'],16562);self.assertEqual(sha(HISTORY),r['index']['sha256']);self.assertEqual(r['new_Final'],0)
  ds=[v['result']['actor']['distance_m'] for v in r['near_neighbor_regression']]
  for d in [3.6356264669372553,4.595351322041442]:self.assertTrue(any(abs(x-d)<1e-6 for x in ds))
 def test_no_mutation_calls(self):
  forbidden={'spawn','spawn_actor','try_spawn_actor','apply_batch','apply_batch_sync','set_simulate_physics','set_transform','apply_control','apply_physics_control','load_world','tick','set_weather'}
  for p in Path('tools').glob('m8h0_*.py'):
   tree=ast.parse(p.read_text());called={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)};self.assertFalse(called&forbidden,(p,called&forbidden))
 def test_frozen_validator_equal(self):
  r=report('frozen_validator_replay');self.assertEqual(r['count'],161);self.assertEqual(r['same_acceptance_count'],161);self.assertEqual(r['new_acceptance_overrides'],0)
 def test_environment_readback(self):
  r=report('map_identity_and_runtime');self.assertEqual(r['vehicles']+r['walkers']+r['sensors'],0);self.assertFalse(r['synchronous_mode']);self.assertEqual(r['physics_changes'],0)
 def test_no_blind_split(self):
  for t in report('curated_template_candidates')['templates']:self.assertFalse(t['independent_split_eligible']);self.assertEqual(t['cone'],'StaticFacilityPlacementNotReady');self.assertEqual(t['barrier'],'StaticFacilityPlacementNotReady')
 def test_fixed_templates_not_replaced_after_warning(self):
  d=report('curated_template_candidates');self.assertTrue(all(not w['replacement_after_review'] for w in d['occlusion_warnings']))
if __name__=='__main__':unittest.main()
