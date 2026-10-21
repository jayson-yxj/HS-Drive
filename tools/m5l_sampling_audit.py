"""Per Actor-view raw silhouette -> reliable labels -> phased feature rays."""
import json
from pathlib import Path
import numpy as np
from tools.m5l_common import rows,write,asset
from datasets.m5f_pixels import homogeneous
from geometry.reference import unproject_pixels_to_optical,transform_points
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import file_hash

def main():
 records=[];grid=VoxelGrid();yy,xx=np.mgrid[:384,:640];splits={}
 for split in ['train','val','regression']:
  for row in rows(split):
   frame=json.loads(Path(row['raw_frame']).read_text());root=Path(row['raw_frame']).parent.parent;cal=json.loads((root/'calibration.json').read_text())['cameras'];labels=np.load(row['pixel_label'])
   for ci,cam in enumerate(row['camera_names']):
    with np.load(root/frame['cameras'][cam]['instance'])as z:sem=z['semantic'];ids=z['instance']
    dep=labels['depth'][ci];interior=homogeneous(sem,ids);patch=np.stack([np.roll(dep,(dy,dx),(0,1))for dy in [-1,0,1]for dx in [-1,0,1]]);strict=np.isfinite(patch).all(0)&((patch.max(0)-patch.min(0))<=.5);points=transform_points(unproject_pixels_to_optical(xx,yy,dep,cal[cam]['K'],'optical_z'),cal[cam]['T_hs_ego_from_opt']);_,inside=grid.locate(points);inside=inside.reshape(dep.shape)
    for actor in row['actors'][ci]['actors']:
     typ='vehicle'if actor['type_id'].startswith('vehicle.')else'person'if actor['type_id'].startswith('walker.')else'cone';accepted=[c['render_id']for c in actor['identity_candidates']if c['accepted']];candidate=[c['render_id']for c in actor['identity_candidates']];raw=np.isin(ids,accepted)&np.isin(sem,[14,15,16,17,18,19]if typ=='vehicle'else[12,13]if typ=='person'else[21]);assoc=labels['actor'][ci]==actor['local_id'];v,u=np.where(raw);box=[int(u.min()),int(v.min()),int(u.max()),int(v.max())]if len(u)else None
     for band,lo,hi in [('all',1.5,80),('20_35',20,35)]:
      distance=(dep>=lo)&(dep<hi);native=raw&distance;reliable=native&labels['valid'][ci];linked=assoc&distance;valid=linked&strict;bev=valid&inside;phase=np.zeros((8,8),int)
      for py in range(8):
       for px in range(8):phase[py,px]=int(bev[py::8,px::8].sum())
      record=dict(split=split,group=row['group'],sample_id=row['sample_id'],camera=cam,actor_id=actor['actor_id'],local_id=actor['local_id'],type=typ,type_id=actor['type_id'],radial_actor_m=actor['distance_m'],band=band,bbox=box,width=box[2]-box[0]+1 if box else 0,height=box[3]-box[1]+1 if box else 0,raw_visible=int(native.sum()),raw_interior=int((native&interior).sum()),reliable_internal=int(reliable.sum()),associated=int(linked.sum()),strict_depth=int(valid.sum()),inside=int(bev.sum()),unassociated_candidate_ids=candidate if not accepted else[],p8_phase_counts=phase.tolist(),strides={str(s):dict(raw=int(native[s//2::s,s//2::s].sum()),internal=int(reliable[s//2::s,s//2::s].sum()),associated=int(linked[s//2::s,s//2::s].sum()),strict=int(valid[s//2::s,s//2::s].sum()),inside=int(bev[s//2::s,s//2::s].sum()))for s in [16,8,4]})
      records.append(record)
   print(split,row['sample_id'],flush=True)
  rr=[r for r in records if r['split']==split and r['band']=='20_35'];bytype=[]
  for typ in ['vehicle','person','cone']:
   sub=[r for r in rr if r['type']==typ];ph=sum((np.array(r['p8_phase_counts'])for r in sub),np.zeros((8,8),int));bytype.append(dict(type=typ,**{k:sum(r[k]for r in sub)for k in ['raw_visible','raw_interior','reliable_internal','associated','strict_depth','inside']},p8_phases=ph.tolist(),strides={str(s):{k:sum(r['strides'][str(s)][k]for r in sub)for k in ['raw','internal','associated','strict','inside']}for s in [16,8,4]},p4_positions=len({r['group']for r in sub if r['strides']['4']['inside']>0}),actor_views_visible=sum(r['raw_visible']>0 for r in sub)))
  p4={r['type']:r['strides']['4']['inside']for r in bytype};positions=len({r['group']for r in rr if r['strides']['4']['inside']>0});splits[split]=dict(types=bytype,p4_gate=dict(passed=sum(p4.values())>=100 and positions>=4 and sum(n>=20 for n in p4.values())>=2,counts=p4,positions=positions))
 # Exact historical support reproduction, no relaxation.
 for split,expected in [('train',335),('val',326)]:
  r=next(r for r in splits[split]['types']if r['type']=='person');assert r['associated']==expected,(split,r);assert r['strides']['8']['inside']==0
 raw='runs/m5l_sampling_actor_trace.json';write(raw,dict(records=records));write('reports/m5l_sampling_failure_analysis.json',dict(kind='offline_actual312CARLAframes_perActor_view',splits=splits,raw=asset(raw),frames=312,p4_training_allowed=all(splits[s]['p4_gate']['passed']for s in ['train','val']),source_sha256=file_hash(__file__),limits='Raw silhouette assigned only viaaccepted renderidentity/AABB association, unassociated candidate IDs retained not silently zeroabsence. Width/height allvisible silhouette; depthbands opticalZ, actor radial separately. Allphases diagnostics noGTROI inference. P4 cannotrecover pixelsremoved byreliability. Correlated repeatedlayouts not independentrays.'))
if __name__=='__main__':main()
