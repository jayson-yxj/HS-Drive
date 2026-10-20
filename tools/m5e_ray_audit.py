"""Per-ray rejection audit and Candidate geometry, never used as absence GT."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from datasets.m5b_pairs import camera_rays
from datasets.m5e_visibility import planar_patch,patches
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default='/root/gpufree-data/hs-drive-data/processed/m5d_index');p.add_argument('--prefix',default='m5e_witness_attrition');p.add_argument('--split',default='val');a=p.parse_args()
    dest=Path('reports',a.prefix+'.json');art=Path('reports',a.prefix+'_rays');art.mkdir(exist_ok=False)
    rows=json.loads(Path(a.index,'index.json').read_text())['samples'];records=[];grid=VoxelGrid()
    for group in sorted({r['group'] for r in rows if r['split']==a.split}):
        rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['configuration']);raw=Path(rr[0]['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());empty=json.loads(Path(rr[0]['raw_frame']).read_text());opposite={n:camera_rays(raw,man,empty,n) for n in empty['cameras']}
        with np.load(rr[0]['label']) as z:la={k:z[k] for k in z.files}
        for row in rr[1:-1]:
            f=json.loads(Path(row['raw_frame']).read_text());allr=[]
            with np.load(row['label']) as z:lb={k:z[k] for k in z.files}
            for cam in f['cameras']:
                r,rd=camera_rays(raw,man,f,cam);o,od=opposite[cam];ids,inside=grid.locate(r['endpoints']);rad=np.linalg.norm(r['endpoints'][:,:2],axis=-1);actor=np.full(len(ids),-1,int);types={}
                for ac in f['actors']:
                    if ac['is_ego']:continue
                    box=np.array(ac['corners_hs_ego_m']);hit=((r['endpoints']>=box.min(0)-.05)&(r['endpoints']<=box.max(0)+.05)).all(-1);aid=int(ac.get('id',ac.get('actor_id',0)));actor[hit]=aid;types[str(aid)]=ac.get('type_id')
                m=inside&r['reliable']&np.isin(r['semantics'],[2,3])&(actor>=0)&(rad>=10)&(rad<35)
                uv=r['uv'][m];u,v=uv.T;rank=ids[m];points=r['endpoints'][m]
                if not len(uv):continue
                dp=patches(od[0],uv);sp=patches(od[1],uv);ip=patches(od[2],uv);fit=planar_patch(dp,sp,ip)
                oldmap=np.zeros(od[0].shape,bool);ou,ov=o['uv'].T;oldmap[ov,ou]=o['reliable']&np.isin(o['semantics'],[1,2,3,4])
                stable=o['stable_mask'][v,u]&~o['self_mask'][v,u];inrange=np.isfinite(od[0][v,u])&(od[0][v,u]>=1.5)&(od[0][v,u]<80)
                behind=cv2.erode(od[0],np.ones((3,3),np.uint8))[v,u]>rd[0][v,u]+.5
                semantic_safe=np.ones(len(uv),bool)
                for lab in [la,lb]:
                    for k in ['ego_mask','semantic_conflict_mask','evidence_conflict_mask']:semantic_safe&=~lab[k].ravel()[rank]
                semantic_safe&=~la['surface_semantic'][1].ravel()[rank]
                candidate=fit['candidate']&stable&inrange&behind&semantic_safe&~oldmap[v,u]
                why=np.zeros(len(uv),np.uint16)
                for bit,bad in enumerate([~stable,~inrange,~behind,~oldmap[v,u],~semantic_safe,~fit['identity_uniform'],fit['residual_m']>.05]):why|=bad.astype(np.uint16)<<bit
                path=art/f"{group}_{row['configuration']}_{cam}.npz"
                np.savez_compressed(path,uv=uv,camera=np.array(cam),frame_id=np.array(f['frame_id']),depth_present=rd[0][v,u],depth_background=od[0][v,u],background_patch=dp,background_semantic_patch=sp,background_instance_patch=ip,present_semantic=rd[1][v,u],actor_id=actor[m],points_hs=points,coarse_rank=rank,distance=rad[m],rejection_bits=why,old_local_reliable=oldmap[v,u],candidate=candidate,plane_residual_m=fit['residual_m'],depth_patch_range_m=fit['range_m'],mixed_present=lb['mixed_cell_mask'].ravel()[rank],mixed_absent=la['mixed_cell_mask'].ravel()[rank])
                rec=dict(group=group,configuration=row['configuration'],camera=cam,path=str(path),sha256=file_hash(path),n=len(uv),actors=types,by_distance={})
                for lo,hi in [(10,15),(15,20),(20,25),(25,35)]:
                    band=(rad[m]>=lo)&(rad[m]<hi);rej=band&~oldmap[v,u]
                    rec['by_distance'][f'{lo}_{hi}']=dict(n=int(band.sum()),old_local_rejected=int(rej.sum()),continuous_plane_candidate=int((rej&fit['candidate']).sum()),identity_edge=int((rej&~fit['identity_uniform']).sum()),nonplanar_or_noise=int((rej&fit['identity_uniform']&~fit['candidate']).sum()),qualified_candidate_pixels=int((band&candidate).sum()),semantic_voxel_veto=int((band&~semantic_safe).sum()))
                records.append(rec)
            print(group,row['configuration'],flush=True)
    summary={band:{k:sum(r['by_distance'][band][k] for r in records) for k in records[0]['by_distance'][band]} for band in ['10_15','15_20','20_25','25_35']}
    atomic_json(dest,dict(kind='offline_perray_actual_CARLA_renderer_audit',records=records,summary=summary,source_sha256=file_hash(__file__),candidate_source_sha256=file_hash('datasets/m5e_visibility.py'),candidate_used_as_negative=False,reason_bit_order=['reference_unstable_or_self','counterfactual_out_of_range','counterfactual_not_behind','old_local_depth_rejected','coarse_semantic_or_ego_veto','different_identity_or_semantic_patch','plane_residual_gt_5cm'],limits='Plane-consistency supports continuous-slope explanation; same-renderer candidate not independent true-free guarantee. Nonplanar/noisy cannot uniquely separate noise from real curvature. Pose/sync invariants from capture; no inter-camera triangulation proof. Every reliable visible target ray10-35 retained, including rejected/candidate. No wholecell absence claim.'))


if __name__=='__main__':main()
