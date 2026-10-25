"""Offline diagnostics of every already-frozen candidate; zero additional scan queries."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from tools.m8g_common import *
from datasets.m8g_geometry import planned_box,inside_box,OpenDriveGeometry

def main():
 scan=json.loads((ROOT/'scan_all_candidates.json').read_text());inventory=json.loads((ROOT/'live_inventory.json').read_text());od=OpenDriveGeometry((ROOT/'Town02.xodr').read_text());rows=[]
 for r in scan['records']:
  phases={}
  for phase,req in r['requests'].items():
   box=planned_box(req);where=r['phases'][phase];lanes=[];overhead=[]
   for i,(rr,lane) in enumerate(zip(req['rays'],req['lane_checks'])):
    try:g=od.point(where['road_id'],where['lane_id'],where['s'],lane['point'])
    except (ValueError,KeyError) as ex:g=dict(error=str(ex),within=False)
    lanes.append(dict(sample=i,corner=i in [0,2,12,14],SDK=lane,OpenDRIVE=g))
    selected=rr['decision'].get('selected')
    for h in rr['hits']:
     if selected and h['semantic'] not in req['declared_region']['semantics'] and h['xyz'][2]>selected['xyz'][2]+.1:
      sources=[o for o in inventory['environment']+inventory['actors'] if inside_box(h['xyz'],o)]
      overhead.append(dict(sample=i,hit=h,height_above_ground_m=h['xyz'][2]-selected['xyz'][2],clearance_to_top_m=None if box is None else h['xyz'][2]-box['top_z'],within_planned_body=bool(box and inside_box(h['xyz'],box,0)),sources=[dict(entity_id=o['entity_id'],name=o['name'],tags=o['semantic_tags'],geometry='ConservativeOBB_NotMesh') for o in sources],render_confirmation='NotRun_NoNewRGB',status='Unresolved_NoOverride'))
   phases[phase]=dict(hypothetical_box=box,lane_points=lanes,overhead=overhead,static=req['static_competitors'],status=req['status'],reasons=req['reasons'],unknown_not_waived=True)
  rows.append(dict(key=r['key'],phases=phases))
 write('reports/m8g_scan_diagnostics.json',dict(**provenance('OfflineFixed80CandidateDiagnostic_NoNewQueries'),scan=asset(ROOT/'scan_all_candidates.json'),rows=rows,acceptance_changes=0))
 print('diagnosed',len(rows),'fixed candidates',flush=True)
if __name__=='__main__':main()
