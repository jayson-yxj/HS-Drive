"""Offline comparison with frozen M8F rules using stored native-point queries."""
import json,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tools.m8h0_common import *
from simulator.m8f_placement import validate_request

def main():
 import carla
 contract=json.loads(Path('configs/m8f/contract.json').read_text());env=json.loads((ROOT/'environment_inventory.json').read_text())['objects'];rows=[]
 for file in ['vehicle_inventory_full.json','navigation_inventory_full.json']:
  for r in json.loads((ROOT/file).read_text())['records']:
   if not r.get('transform'):continue
   g=r['geometry'];p=r['transform'];t=carla.Transform(carla.Location(p['xyz'][0],p['xyz'][1],g['expected_map_z_m']),carla.Rotation(pitch=p['rotation'][0],yaw=p['rotation'][1],roll=p['rotation'][2]))
   with patch('simulator.m8f_placement.ray',side_effect=copy.deepcopy(g['rays'])),patch('simulator.m8f_placement.lane_check',side_effect=copy.deepcopy(g['lane_checks'])):
    frozen=validate_request(None,None,SimpleNamespace(transform=t),r['asset'],contract,env)
   supplemental=set(g['reasons'])-set(frozen['reasons']);same=(frozen['status']=='GroundedByGeometryProxy')==(g['status']=='GeometryCandidateProxySupported')
   allow=supplemental<={'MissingGroundEvidence','DeclaredRegionUnsupported','NativeHeadingDisagreesWithDeclaredLane','StaticGeometryOverlap','SupportHeightDisagreement','BottomSupportGap'}
   assert set(frozen['reasons'])<=set(g['reasons']) and allow,r['id']
   assert not(g['status']=='GeometryCandidateProxySupported' and frozen['status']!='GroundedByGeometryProxy')
   box_match=None
   if frozen['requested_obb'] is not None:
    box_match=bool(np.allclose(frozen['requested_obb']['corners_world_m'],g['hypothetical_vertical_only_box']['corners_world_m'],atol=1e-5));assert box_match
   rows.append(dict(id=r['id'],frozen_status=frozen['status'],native_status=g['status'],same_acceptance=same,frozen_reasons=frozen['reasons'],native_reasons=g['reasons'],supplemental_rejections=sorted(supplemental),OBB_match=box_match,all_stored_queries_replayed=True,acceptance_relaxed=False))
 write('reports/m8h0_frozen_validator_replay.json',dict(**provenance('OfflineFrozenValidatorReplay_NoNewCARLAQueries'),records=rows,count=len(rows),same_acceptance_count=sum(r['same_acceptance'] for r in rows),new_acceptance_overrides=0,replay_scope='Recorded ray and lane answers at preserved native XY/yaw; no simulator calls, no validation of random navigation regeneration',frozen_code=[asset('simulator/m8f_placement.py'),asset('datasets/m8f_quality.py')]))
 print('frozen full-validator replay',len(rows),'no acceptance relaxation',flush=True)
if __name__=='__main__':main()
