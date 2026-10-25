"""Review records bind frozen templates to precise package evidence without resampling."""
import json
from pathlib import Path
from tools.m8h0_common import *
def main():
 d=json.loads(Path('reports/m8h0_curated_template_candidates.json').read_text());rows=[]
 notes=[
  'Vehicle along Road11; person on Road9 sidewalk viewed rear-right. Lamp OBB center-view warning for vehicle. No point moved.',
  'Both targets on Road19. Person center-view intersects lamp and bus-stop OBB proxies. Ego is near old M8F/G Development01; never blind.',
  'Both targets on Road0, front/rear view. Vehicle lamp-box warning; pedestrian rear-left/rear rays have no OBB warning. Same historical corridor.',
  'Ego Road8, targets Road11 in rear-right view. Vehicle lamp-box warning. Source is a multi-road observation template, not an asserted straight-route driving scene.'
 ]
 for t,n in zip(d['templates'],notes):
  rows.append(dict(template=t['id'],review='Agent review of stored numeric geometry and plotted map, not human safety approval',notes=n,template_status='CuratedTemplateCandidateGo',full_geometry='TemplateGeometryPartial',actual_render='NotRun',initial_native_transforms_unchanged=True,proposed_next_smoke=t['id'] in ['m8h0_template_00','m8h0_template_02'],no_execution_authorized=True))
 write('reports/m8h0_template_review.json',dict(**provenance('OfflineFixedTemplateCuration_NoExtraQueries'),templates=asset('reports/m8h0_curated_template_candidates.json'),map_binding=asset('reports/m8h0_map_identity_and_runtime.json'),environment_binding=asset(ROOT/'environment_inventory.json'),rig_binding=asset('configs/cameras.yaml'),uniform_geometry_contract=asset('configs/m8f/contract.json'),identity_contract=asset('configs/m8f/contract.json'),history_binding=asset('reports/m8h0_historical_preflight.json'),figures=asset('reports/m8h0_visualizations.json'),rows=rows,replacement_after_review=False,limits='No new map or ray calls. No physical mesh, actual Actor settling or render identity certification.'))
if __name__=='__main__':main()
