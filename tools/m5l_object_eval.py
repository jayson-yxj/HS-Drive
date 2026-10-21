"""2.5D conditional flat plane vs Dense depth on identical RGB components."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from tools.m5l_common import rows,cache_index,write,asset,summary
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5k_geometry import resample_probability
from models.m5l_p4 import sample_map
from diagnostics.m5l_objects import components,locate,match
from training.runtime import prepare_cuda
from simulator.recorder import file_hash

def errors(p,ref):
 delta=p[:2]-ref[:2];a=math.atan2(p[1],p[0]);b=math.atan2(ref[1],ref[0]);return dict(xy_m=float(np.linalg.norm(delta)),radial_error_m=float(abs(np.linalg.norm(p[:2])-np.linalg.norm(ref[:2]))),bearing_error_deg=abs(math.degrees(math.atan2(math.sin(a-b),math.cos(a-b)))))

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);args=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();ix=cache_index(args.split);records=[];detections=[]
 for row in rows(args.split):
  x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);z=np.load(ix[row['sample_id']]);frame=json.loads(Path(row['raw_frame']).read_text());pose=np.array(frame['T_world_from_ego']);pitch=math.degrees(math.asin(np.clip(pose[2,0],-1,1)))
  for seed in [101,102]:
   q=z[f'pixel_{seed}'].astype(np.float32)
   for ci,cam in enumerate(row['camera_names']):
    cand=components(q[ci]);truth=[];meta=[]
    for a in row['actors'][ci]['actors']:
     m=d['actor'][ci]==a['local_id']
     if m.sum()>=3:truth.append(m);meta.append(a)
    pairs,fp,fn=match(cand,truth);merges=sum(sum((cc['mask']&t).sum()>=3 for t in truth)>1 for cc in cand);splits=sum(sum((cc['mask']&t).sum()>=3 for cc in cand)>1 for t in truth);K=x['intrinsics'][0,ci].cpu().numpy();T=x['cam_to_ego'][0,ci].cpu().numpy();locations=[locate(cc,K,T)for cc in cand]
    detections.append(dict(group=row['group'],sample_id=row['sample_id'],seed=seed,camera=cam,truth=len(truth),candidates=len(cand),matched=len(pairs),unmatched=len(fn),unmatched_candidates=len(fp),merged_components=int(merges),split_actors=int(splits),abstained=sum(v[0]is None for v in locations),flat_pose=abs(pitch)<=1,ego_pitch_deg=pitch,configuration=row['configuration']))
    pairmap={j:i for i,j,_ in pairs}
    for j,a in enumerate(meta):
     typ='vehicle'if a['type_id'].startswith('vehicle.')else'person'if a['type_id'].startswith('walker.')else'cone';base=dict(group=row['group'],sample_id=row['sample_id'],seed=seed,camera=cam,actor_id=a['actor_id'],type=typ,distance_m=a['distance_m'],detected=j in pairmap,ego_pitch_deg=pitch,flat_pose=abs(pitch)<=1,configuration=row['configuration']);m=truth[j];vv,uu=np.where(m);bottom=int(vv.max());pixels=m&(np.indices(m.shape)[0]>=bottom-1);v,u=np.where(pixels);dp=d['depth'][ci,v,u];dirs=np.linalg.solve(K,np.stack([u,v,np.ones(len(u))]));pts=(T[:3,:3]@(dirs*dp)+T[:3,3,None]).T;contact=pts.mean(0);center=np.array(a['center']);base.update(visible_bottom_z=float(contact[2]),contact_matches_plane=abs(contact[2]+.25)<=.15,reference='AABB XYcenter proxy and visible rendererbottomendpoint; no exactphysicalcontact',visible_pixels=int(m.sum()))
     if j not in pairmap:records.append(base|dict(valid=False,reason='undetected'));continue
     ci2=pairmap[j];cc=cand[ci2];point,reason=locations[ci2];uv=cc['uv'];pr=sample_map(c['probs'][0,ci:ci+1],torch.tensor(uv[None,None],device='cuda'),stride=8,center=4)[0,0];pr=pr/pr.sum();depths={'dense_expected':float((pr*net.splat.depth_centers).sum()),'dense_mode':float(net.splat.depth_centers[pr.argmax()])};base.update(valid=point is not None,reason=reason,uv=uv.tolist())
     if point is not None:base['plane_center']=errors(point,center);base['plane_bottom']=errors(point,contact)
     ray=T[:3,:3]@np.linalg.solve(K,np.array([*uv,1.]))
     for name,depth in depths.items():
      pred=T[:3,3]+depth*ray;base[name+'_center']=errors(pred,center);base[name+'_bottom']=errors(pred,contact)
     records.append(base)
  print(row['sample_id'],flush=True)
 raw=f'runs/m5l_objects_{args.split}.json';write(raw,dict(actors=records,detections=detections));out=[]
 for seed in [101,102]:
  for scope in ['all','flat_pose','nonflat_pose','contact_supported','vehicle','person','cone']:
   rr=[r for r in records if r['seed']==seed and (scope=='all'or scope=='flat_pose'and r['flat_pose']or scope=='nonflat_pose'and not r['flat_pose']or scope=='contact_supported'and r['flat_pose']and r['contact_matches_plane']or scope==r['type'])];valid=[r for r in rr if r['valid']];det=[r for r in rr if r['detected']];out.append(dict(seed=seed,scope=scope,actor_views=len(rr),detected=len(det),valid=len(valid),positions=len({r['group']for r in valid}),valid_given_detected=len(valid)/max(len(det),1),errors={method:summary([r[method]['xy_m']for r in valid])for method in ['plane_center','plane_bottom','dense_expected_center','dense_expected_bottom','dense_mode_center','dense_mode_bottom']}))
 write(f'reports/m5l_objects_{args.split}.json',dict(kind='actual_GPU_offline_RGB_components_conditional2p5D',aggregate=out,detection=[dict(seed=s,**{k:sum(r[k]for r in detections if r['seed']==s)for k in ['truth','candidates','matched','unmatched','unmatched_candidates','merged_components','split_actors','abstained']})for s in [101,102]],raw=asset(raw),source_sha256=file_hash(__file__),limits='Plane z=-.25 fixed engineeringassumption, noGTroadinput. Binaryforeground cannotconfirmgroundcontact/occlusion; valid outputs conditionalhypotheses NOT trustedpositions. contact_supported/flat_pose areGTdiagnosticstrata neverpredictiongates. AABBcenter approximation vs visiblecontact offset explicit. SameRGBcomponent/bottomray forDenseExpected/Mode baseline; not oracleGTROI. 1to1IoU>=.1 reliablevisiblemask; unmatchedcandidates may unknown not allphysicalfalseobjects.'))
if __name__=='__main__':main()
