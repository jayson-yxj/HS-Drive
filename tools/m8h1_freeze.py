"""Historical seal and exact approved template binding before actor creation."""
import json
from pathlib import Path
from tools.m8h1_common import *
from tools.m8e_verify import verify

def main():
 n=verify('reports/m8h0_release_manifest.json');old=json.loads(Path('reports/m8h0_curated_template_candidates.json').read_text());cfg=json.loads(Path('configs/m8h1/protocol.json').read_text());groups=[t for t in old['templates'] if t['id'] in cfg['templates']]
 assert len(groups)==2 and [(g['ego']['id'],g['vehicle']['id'],g['person']['id']) for g in groups]==[('spawn_044','spawn_049','nav_07'),('spawn_100','spawn_045','nav_33')]
 meta=json.loads(Path('reports/m8h0_map_identity_and_runtime.json').read_text())
 for a in meta['packaged_files']:assert sha(a['path'])==a['sha256'],a['path']
 assert sha('configs/cameras.yaml')==json.loads(Path('reports/m8h0_template_review.json').read_text())['rig_binding']['sha256']
 write('reports/m8h1_frozen_templates.json',dict(**provenance('ReadOnlyFreezeBeforeActualActor'),templates=groups,M8H0_source=asset('reports/m8h0_curated_template_candidates.json'),M8H0_seal=asset('reports/m8h0_release_manifest.json'),sealed_files_verified=n,map_identity=asset('reports/m8h0_map_identity_and_runtime.json'),rig=asset('configs/cameras.yaml'),original_geometry_contract=asset('configs/m8f/contract.json'),history=asset(HISTORY),split='DevelopmentOnly',history_coverage='HistoricalCoverageIncomplete',FinalEligible=False,no_new_navigation_calls=True))
 print('Frozen exact00/02, map/rig/packageSHA verified',flush=True)
if __name__=='__main__':main()
