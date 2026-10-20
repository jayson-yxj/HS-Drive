"""Independent positive target support and reliable depth, never candidate negatives."""
import argparse,json
from pathlib import Path
import numpy as np
from datasets.m5b_pairs import camera_rays
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);a=p.parse_args();root=Path(a.index);dest=root/'m5e_supervision.json'
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads((root/'index.json').read_text())['samples'];records=[];g=VoxelGrid()
    for row in rows:
        raw=Path(row['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());f=json.loads(Path(row['raw_frame']).read_text());arrays={};positive=np.zeros(np.prod(g.shape),np.int32);camera_counts=[]
        with np.load(row['label']) as z:lab={k:z[k] for k in z.files}
        safe=lab['obstacle_valid']&lab['surface_semantic_valid']&lab['surface_semantic'][1]&~lab['surface_semantic'][0]&~lab['surface_semantic'][2]&~lab['ego_mask']&~lab['evidence_conflict_mask']&~lab['semantic_conflict_mask']
        for name in row['camera_names']:
            r,d=camera_rays(raw,man,f,name);rank,inside=g.locate(r['endpoints']);u,v=r['uv'].T;candidate=np.zeros(len(u),bool)
            for ac in f['actors']:
                if ac['is_ego']:continue
                b=np.array(ac['corners_hs_ego_m']);candidate|=((r['endpoints']>=b.min(0)-.05)&(r['endpoints']<=b.max(0)+.05)).all(-1)
            reliable=r['reliable'];target=reliable&candidate&np.isin(r['semantics'],[2,3]);train=target&inside&((u+2*v)%4!=1);np.add.at(positive,rank[train],1)
            densevalid=np.zeros(d[0].shape,bool);densevalid[v,u]=reliable
            densetarget=np.zeros_like(densevalid);densetarget[v,u]=target
            rad=np.linalg.norm(r['endpoints'][:,:2],axis=-1)
            camera_counts.append(dict(camera=name,visible_target_pixels=int(target.sum()),by_distance={f'{lo}_{hi}':int((target&(rad>=lo)&(rad<hi)).sum()) for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]}))
            for stride in [16,8]:
                vv,uu=np.meshgrid(np.rint((np.arange(384//stride)+.5)*stride-.5).astype(int),np.rint((np.arange(640//stride)+.5)*stride-.5).astype(int),indexing='ij')
                for key,value in [('depth',d[0][vv,uu]),('valid',densevalid[vv,uu]),('actor',densetarget[vv,uu])]:arrays.setdefault(f'{key}{stride}',[]).append(value)
        arrays={k:np.array(v) for k,v in arrays.items()};arrays['presence']=(positive.reshape(g.shape)>=3)&safe
        path=root/row['group']/f"m5e_supervision_{f['frame_id']}.npz";np.savez_compressed(path,**arrays)
        z,x,y=np.where(arrays['presence']);dist=np.hypot(-20+(x+.5)*.5,-30+(y+.5)*.5)
        records.append(dict(sample_id=row['sample_id'],group=row['group'],split=row['split'],configuration=row['configuration'],path=str(path),sha256=file_hash(path),presence_cells=int(arrays['presence'].sum()),by_distance={f'{lo}_{hi}':int(((dist>=lo)&(dist<hi)).sum()) for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]},cameras=camera_counts));print(row['sample_id'],records[-1]['presence_cells'],flush=True)
    summary={}
    for split in ['train','val','test']:
        rr=[r for r in records if r['split']==split];summary[split]={b:dict(presence_cells=sum(r['by_distance'][b] for r in rr),groups=sorted({r['group'] for r in rr if r['by_distance'][b]}),visible_pixels=sum(c['by_distance'][b] for r in rr for c in r['cameras'])) for b in ['0_10','10_15','15_20','20_25','25_35','35_80']}
    atomic_json(dest,dict(records=records,summary=summary,source_sha256=file_hash(__file__),candidate_used=False,contract='Presence uses >=3 reliable nonheld rendered Actor surface endpoints and safeV2positive. Depth samples reliable same center convention, no GT input. Sampled depth atfeatureUV can overlap heldpixel class but val/test are separategroups; absolute reference same-renderer not independent.'))


if __name__=='__main__':main()
