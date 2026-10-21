"""Oracle2D/Depth factorization, matched UV and mask; oldBEV kept as separate baseline."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from diagnostics.m5c_localization import matching
from diagnostics.m5d_metrics import voxel_3d_matching
from diagnostics.m5f_metrics import histogram,metrics
from tools.m5f_common import inputs,pixel_label
from tools.m5f_dataset import ROOT
from tools.m5a_learning import encoder_load
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5d_stride8 import cache_p8
from models.m5e_paths import geometry
from models.m5e_paths import depth_logits
from tools.m5e_common import load_checkpoint
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5f_lift import lift
from geometry.ray_tracing import VoxelGrid
from geometry.reference import unproject_pixels_to_optical,transform_points
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def reference(d,inp,rows,cameras):
    grid=VoxelGrid();size=np.prod(grid.shape);pos=np.zeros(size,np.int32);neg=pos.copy();actor={};v,u=np.mgrid[:384,:640]
    for c in range(6):
        pts=transform_points(unproject_pixels_to_optical(u,v,d['depth'][c],inp['intrinsics'][0,c].cpu().numpy(),'optical_z'),inp['cam_to_ego'][0,c].cpu().numpy());r,inside=grid.locate(pts);r=r.reshape(384,640);inside=inside.reshape(384,640);yes=d['valid'][c]&inside
        np.add.at(pos,r[yes&d['target'][c]],1);np.add.at(neg,r[yes&~d['target'][c]],1)
        for a in rows['actors'][c]['actors']:
            key=a['local_id'];rec=actor.setdefault(key,dict(meta=a,counts=np.zeros(size,np.int32)));m=yes&(d['actor'][c]==key);np.add.at(rec['counts'],r[m],1)
    target=(pos>=3).reshape(grid.shape);valid=((pos>=3)|(neg>=3)).reshape(grid.shape)&~((pos>0)&(neg>0)).reshape(grid.shape)
    actors=[dict(**v['meta'],mask=(v['counts']>=3).reshape(grid.shape)&valid&target) for v in actor.values()]
    return target,valid,actors


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default=ROOT);p.add_argument('--split',default='val');a=p.parse_args();dest=Path(f'reports/m5f_{"factorization_validation" if a.split=="val" else "2d_3d_factorization"}.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);rows=[r for r in json.loads(Path(a.index,'index.json').read_text())['records'] if r['split']==a.split];pix=json.loads(Path(f'reports/m5f_{"pixel_validation" if a.split=="val" else "2d_presence_absence"}.json').read_text());records=[];net=encoder_load();art=Path(f'runs/m5f_{a.split}_bev');art.mkdir(exist_ok=False);thresholds=[.1,.2,.4,.6,.8];frozen=json.loads(Path('reports/m5f_geometry_selection.json').read_text()) if a.split=='test' else None
    # Original depth for all intervention models. Tuned pixel FPN never changes it.
    for row in rows:
        inp=inputs(row);d=pixel_label(row);target,valid,actors=reference(d,inp,row,row['camera_names']);cache={16:cache_rgb(net,inp),8:cache_p8(net,inp)}
        for model in pix['records']:
            rr=next(x for x in model['predictions'] if x['sample_id']==row['sample_id']);prob=np.load(rr['path'])['probability_u8'].astype(np.float32)/256;stride=model['stride'];c=cache[stride];h,w=c['features'].shape[-2:];vv,uu=np.meshgrid(np.rint((np.arange(h)+.5)*stride-.5).astype(int),np.rint((np.arange(w)+.5)*stride-.5).astype(int),indexing='ij')
            depth=torch.from_numpy(d['depth'][:,vv,uu][None]).cuda();mask=torch.from_numpy(d['valid'][:,vv,uu][None]).cuda();gt=torch.from_numpy(d['target'][:,vv,uu][None].astype(np.float32)).cuda();pred=torch.from_numpy(prob[:,vv,uu][None]).cuda();g=geometry(net,c,depth)
            for mode,fg,dep in [('A_oracle2d_oracledepth',gt,depth),('B_oracle2d_learneddepth',gt,None),('C_learned2d_oracledepth',pred,depth),('D_learned2d_learneddepth',pred,None)]:
                out=lift(net.splat,fg,c['probs'],c['K'],c['T'],c['image_hw'],depth=dep,valid=mask);score=out['event_score'][0].cpu().numpy();mass=out['foreground_mass'][0].cpu().numpy();total=out['total_depth_mass'][0].cpu().numpy();key=f"{model['name']}_{model['seed']}_{mode}";folder=art/key;folder.mkdir(exist_ok=True);path=folder/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,score=score,mass=mass,total=total,valid=valid,target=target)
                hist=histogram(score,target,valid);th=.2 if frozen is None else frozen[key]['threshold'];pbin=score>=int(th*256)/256;am=[]
                for actor in actors:
                    m=actor['mask'];n=int(m.sum());sp=matching(pbin&valid,m);am.append(dict(local_id=actor['local_id'],type_id=actor['type_id'],distance_m=actor['distance_m'],visible_cells=n,exact_hits=int((m&pbin).sum()),matching=sp))
                ray=(gt.bool()&mask&g['inside']);mass1=(c['probs']*g['same_layer']*(g['xy']<=1)).sum(2);records.append(dict(name=model['name'],seed=model['seed'],stride=stride,mode=mode,group=row['group'],configuration=row['configuration'],sample_id=row['sample_id'],path=str(path),sha256=file_hash(path),histogram=hist.tolist(),threshold=th,metrics=metrics(hist,th),matching=matching(pbin&valid,target&valid),voxel3d=voxel_3d_matching(pbin&valid,target&valid),actors=am,unknown_high_fraction=float(pbin[~valid].mean()),actor_feature_rays=int(ray.sum()),gt1m_probability_mass=float(mass1[ray].mean()) if ray.any() else None,by_camera=[dict(camera=name,actor_feature_rays=int(ray[0,c].sum()),gt1m_probability_mass=float(mass1[0,c][ray[0,c]].mean()) if ray[0,c].any() else None) for c,name in enumerate(row['camera_names'])]))
            # Unmasked learned candidate, legal RGB-only. GT above only diagnosticmask.
            full=lift(net.splat,pred,c['probs'],c['K'],c['T'],c['image_hw']);path=art/(f"{model['name']}_{model['seed']}_full_{row['sample_id'].replace('/','_')}.npz");np.savez_compressed(path,**{k:v[0].cpu().numpy() for k,v in full.items()},target=target,valid=valid)
        print(row['sample_id'],flush=True)
    if a.split=='val':
        choices={}
        for key in sorted({f"{r['name']}_{r['seed']}_{r['mode']}" for r in records}):
            rr=[r for r in records if f"{r['name']}_{r['seed']}_{r['mode']}"==key];hh=sum(np.array(r['histogram']) for r in rr);curve=[dict(threshold=t,**metrics(hh,t)) for t in thresholds];best=max(curve,key=lambda v:((v['iou'] or 0),v['threshold']));choices[key]=dict(best,curve=curve)
        atomic_json('reports/m5f_geometry_selection.json',choices)
    atomic_json(dest,dict(kind='actual_GPU_offline_fourway_projection_selfconsistency',records=records,source_sha256=file_hash(__file__),scope=a.split,limits='A/B/C/D matched nativeUV+visiblelabelmask perstride. A sharesrendererreference, notindependentphysicaltruth. Full unmasked D storedseparately, noGTforward. Nativegrid differentS8/S16; targetreferenceallnativepixels vsprojectionfeaturegrid exposes samplingceiling. Thresholdsvalidationonly. Foregroundeventscore=1-exp(-mass), notoccupancyprobability. OldBEV separatecomparison, no oldheadOracleinput.'))

if __name__=='__main__':main()
