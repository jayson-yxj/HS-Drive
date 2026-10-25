"""Read-only geometric counterexamples at historical failed placements; no RGB."""
import json
from pathlib import Path
from tools.m8f_common import *
from simulator.m8f_placement import ray
from datasets.m8f_quality import choose_surface

def main():
 import carla
 c=carla.Client('127.0.0.1',2100);c.set_timeout(60)
 cfg=json.loads(Path('configs/m8f/contract.json').read_text());old=json.loads(Path('reports/m8e_instance_identity_audit.json').read_text())['object_rows'];rows=[]
 for group,ids in [('m8e_development_02',['Actor:717','Actor:715']),('m8e_calibration_00',['Actor:1022'])]:
  objects=[a for a in old if a['group']==group and a['entity_key'] in ids]
  town=objects[0]['map'].split('/')[-1];w=c.load_world(town)
  for a in objects:
   x,y,z=a['center_world_m'];r=ray(w,[x,y,z+8],[x,y,z-20]);decision=choose_surface(r['hits'],z-a['extent_m'][2],[1,2,24],cfg['ground'])
   rows.append(dict(historical_group=group,entity_key=a['entity_key'],map=town,kind='NewActualCARLARayAtHistoricalXY_NotHistoricalRayRecovery',raw=r,decision=decision,unsafe_actor_spawned=False))
 write('reports/m8f_water_multilayer_probe.json',dict(**provenance('ActualCARLAGeometryOnlyCounterexample'),records=rows,new_RGB_frames=0,no_claim_old_ray_reconstruction=True))
 print([(r['entity_key'],r['raw']['hits'],r['decision']['status']) for r in rows])
if __name__=='__main__':main()
