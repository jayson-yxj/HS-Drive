"""Peractor support losses summarized across camera, size and phase (read-only)."""
import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write,asset,summary

def main():
 records=json.loads(Path('runs/m5l_sampling_actor_trace.json').read_text())['records'];out=[]
 for split in ['train','val','regression']:
  for typ in ['vehicle','person','cone']:
   for camera in ['all','front','front_left','front_right','rear_left','rear_right','rear']:
    rr=[r for r in records if r['split']==split and r['type']==typ and r['band']=='20_35'and(camera=='all'or r['camera']==camera)];vis=[r for r in rr if r['raw_visible']>0];n=sum(r['raw_visible']for r in rr);reliable=sum(r['associated']for r in rr);strict=sum(r['strict_depth']for r in rr);inside=sum(r['inside']for r in rr);ph=sum((np.array(r['p8_phase_counts'])for r in rr),np.zeros((8,8),int));out.append(dict(split=split,type=typ,camera=camera,visible_actor_views=len(vis),raw_pixels=n,associated_internal=reliable,strict_pixels=strict,inside_pixels=inside,internal_loss=n-reliable,depth_loss=reliable-strict,outside_loss=strict-inside,width=summary([r['width']for r in vis]),height=summary([r['height']for r in vis]),area=summary([r['raw_visible']for r in vis]),p8_default=int(ph[4,4]),p8_best_fixed_phase=int(ph.max()),best_phase_vu=np.unravel_index(ph.argmax(),ph.shape),p8_zero_phases=int((ph==0).sum()),p4=sum(r['strides']['4']['inside']for r in rr)))
 write('reports/m5l_sampling_support_summary.json',dict(kind='offline_perActor_existing312frame_aggregation',records=out,raw=asset('runs/m5l_sampling_actor_trace.json'),limits='Lossstages includeassociation differences, cannot attribute everypixel toonlyerosion. Counts opticalZ20-35; visibleActorview dimensionsfullsilhouette, notrectarea. Bestphase usesGTdiagnostic only, notchoseninference.'))
if __name__=='__main__':main()
