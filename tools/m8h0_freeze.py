"""Freeze M8G and verify unified historical sources before source queries."""
import json
from pathlib import Path
import numpy as np
from tools.m8h0_common import *
from tools.m8e_verify import verify
from tools.m8g_history import preflight

def main():
 sealed=verify('reports/m8g_release_manifest.json');index=json.loads(HISTORY.read_text());assert len(index['records'])==16562
 assert sha(HISTORY)==json.loads(Path('reports/m8g_historical_preflight_index.json').read_text())['index']['sha256']
 for r in index['historical_sources']:assert sha(r['path'])==r['sha256'],r['path']
 old=json.loads(Path('reports/m8g_historical_preflight_index.json').read_text())['regression'];requests=json.loads(Path('reports/m8f_ray_hit_and_contact_audit.json').read_text())['requests'];regs=[]
 for r in old:
  req=next(x for x in requests if x['group']==r['group'] and x['key']==r['target'])
  p=np.array(req['preflight']['present']['requested_obb']['corners_world_m'])[[0,4,6,2],:2]
  raw=dict(index,records=[x for x in index['records'] if x['index_origin']=='historical_raw_frame_supplement.json']);out=preflight(p,'Town02',raw)
  assert abs(out['actor']['distance_m']-r['actual']['actor']['distance_m'])<1e-6
  regs.append(dict(group=r['group'],target=r['target'],result=out,passed=True))
 write(ROOT/'m8g_freeze.json',dict(**provenance('ReadOnlyHistoricalFreezeBeforeNativeQueries'),M8G=asset('reports/m8g_release_manifest.json'),verified_sealed_assets=sealed,history=asset(HISTORY),history_source_count=len(index['historical_sources']),history_records=16562,regression=regs,closed_state='Partial / PlacementFeasibilityNoGo / TrainingReadinessNoGo',frozen_facts=dict(unresolved_requests=7,NONE_trigger_supported_without_component=21,lane_ambiguity_unapproved=130,geometry_proxy_candidates=28,total_candidates=80,Development00_vehicle=0,RGB_smoke='NotRun',new_Final=0),history_coverage='HistoricalCoverageIncomplete',previous_source_expansion='Stopped'))
 print('M8G',sealed,'sealed assets; all historical source SHA verified; five footprint regressions passed',flush=True)
if __name__=='__main__':main()
