"""Diagnostic rejection funnel on validation; never changes witness selection."""

import json
from pathlib import Path
import cv2
import numpy as np
from datasets.m5b_pairs import camera_rays
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json, file_hash


def main():
    dest=Path('reports/m5d_witness_attrition.json')
    if dest.exists(): raise FileExistsError(dest)
    index=Path('/root/gpufree-data/hs-drive-data/processed/m5d_index')
    rows=json.loads((index/'index.json').read_text())['samples']
    records=[]; grid=VoxelGrid()
    for group in sorted({r['group'] for r in rows if r['split']=='val'}):
        rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['configuration'])
        raw=Path(rr[0]['raw_frame']).parent.parent
        man=json.loads((raw/'manifest.json').read_text()); empty=json.loads(Path(rr[0]['raw_frame']).read_text())
        opposite={n:camera_rays(raw,man,empty,n) for n in empty['cameras']}
        with np.load(rr[0]['label']) as z: la={k:z[k] for k in z.files}
        for row in rr[1:-1]:
            frame=json.loads(Path(row['raw_frame']).read_text())
            with np.load(row['label']) as z: lb={k:z[k] for k in z.files}
            for name in frame['cameras']:
                r,rd=camera_rays(raw,man,frame,name); other,od=opposite[name]
                ids,inside=grid.locate(r['endpoints']); u,v=r['uv'].T
                candidate=np.zeros(len(ids),bool)
                for a in frame['actors']:
                    if a['is_ego']:continue
                    box=np.array(a['corners_hs_ego_m']);candidate|=((r['endpoints']>=box.min(0)-.05)&(r['endpoints']<=box.max(0)+.05)).all(-1)
                initial=inside&r['reliable']&candidate&np.isin(r['semantics'],[2,3])
                om=np.zeros(od[0].shape,bool);ou,ov=other['uv'].T;om[ov,ou]=other['reliable']&np.isin(other['semantics'],[1,2,3,4])
                stages={'visible_target':initial.copy()}
                initial &= other['stable_mask'][v,u]&~other['self_mask'][v,u]&np.isfinite(od[0][v,u])&(od[0][v,u]>=1.5)&(od[0][v,u]<80)
                stages['counterfactual_inrange_stable']=initial.copy()
                initial &= cv2.erode(od[0],np.ones((3,3),np.uint8))[v,u]>rd[0][v,u]+.5
                stages['counterfactual_behind']=initial.copy()
                initial &= om[v,u]
                stages['counterfactual_local_depth_reliable']=initial.copy()
                initial &= rd[2][v,u]!=od[2][v,u]
                clipped=np.clip(ids,0,np.prod(grid.shape)-1)
                for lab in [la,lb]:
                    for key in ['ego_mask','evidence_conflict_mask','semantic_conflict_mask']:initial &= ~lab[key].ravel()[clipped]
                initial &= ~la['surface_semantic'][1].ravel()[clipped]
                stages['voxel_semantic_safe']=initial.copy()
                rad=np.linalg.norm(r['endpoints'][:,:2],axis=1)
                for lo,hi in [(0,10),(10,15),(15,20),(20,35),(35,80)]:
                    band=(rad>=lo)&(rad<hi)
                    records.append(dict(group=group,configuration=row['configuration'],camera=name,distance=f'{lo}_{hi}',counts={k:int((m&band).sum()) for k,m in stages.items()}))
    summary={}
    for band in ['0_10','10_15','15_20','20_35','35_80']:
        chosen=[r for r in records if r['distance']==band]
        summary[band]={k:sum(r['counts'][k] for r in chosen) for k in chosen[0]['counts']}
    atomic_json(dest,dict(kind='offline_new_validation_counterfactual_rejection_funnel',records=records,summary=summary,source_sha256=file_hash(__file__),limits='Ordered diagnostic pixel funnel, prior to dense-obstacle veto and >=3 per-cell vote. Counts are correlated pixels. Does not revise fixed masks or tune on test.'))


if __name__=='__main__':main()
