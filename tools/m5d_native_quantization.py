"""Dense native rendered point quantization, distinct from feature-ray sampling."""

import json
from pathlib import Path
import numpy as np
from datasets.m5b_pairs import camera_rays
from datasets.labels import depth_bin_edges
from geometry.reference import unproject_pixels_to_optical, transform_points
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json, file_hash


def main():
    dest=Path('reports/m5d_native_quantization.json')
    if dest.exists(): raise FileExistsError(dest)
    rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5d_index/index.json').read_text())['samples']
    edges=depth_bin_edges(48,1.5,80);centers=(edges[1:]+edges[:-1])/2;grid=VoxelGrid();out=[]
    for row in rows:
        if row['split']!='val' or row['configuration'] not in [3,4,5,6,9,11,12,14]:continue
        raw=Path(row['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());frame=json.loads(Path(row['raw_frame']).read_text());cal=json.loads((raw/'calibration.json').read_text())['cameras']
        for name in row['camera_names']:
            r,data=camera_rays(raw,man,frame,name);candidate=np.zeros(len(r['uv']),bool)
            for a in frame['actors']:
                if a['is_ego']:continue
                box=np.array(a['corners_hs_ego_m']);candidate|=((r['endpoints']>=box.min(0)-.05)&(r['endpoints']<=box.max(0)+.05)).all(-1)
            rank,inside=grid.locate(r['endpoints']);keep=r['reliable']&candidate&inside&np.isin(r['semantics'],[2,3]);u,v=r['uv'][keep].T;gt=r['endpoints'][keep];depth=data[0][v,u]
            if not len(depth):continue
            nearest=centers[np.abs(depth[:,None]-centers).argmin(1)]
            q=transform_points(unproject_pixels_to_optical(u,v,nearest,cal[name]['K'],'optical_z'),cal[name]['T_hs_ego_from_opt'])
            qr,qi=grid.locate(q);gr=rank[keep];rad=np.linalg.norm(gt[:,:2],axis=1);xy=np.linalg.norm(q[:,:2]-gt[:,:2],axis=1)
            for lo,hi in [(10,15),(15,20),(20,35)]:
                m=(rad>=lo)&(rad<hi)
                if not m.any():continue
                out.append(dict(group=row['group'],configuration=row['configuration'],camera=name,distance=f'{lo}_{hi}',n=int(m.sum()),correct_voxel_fraction=float(((qr==gr)&qi)[m].mean()),xy_mae_m=float(xy[m].mean()),xy_p95_m=float(np.quantile(xy[m],.95)),same_layer_1m=float(((qr//19200==gr//19200)&qi&(xy<=1))[m].mean())))
    summary=[]
    for band in ['10_15','15_20','20_35']:
        rr=[r for r in out if r['distance']==band];n=sum(r['n'] for r in rr)
        summary.append(dict(distance=band,n=n,groups=len({r['group'] for r in rr}),**{k:sum(r[k]*r['n'] for r in rr)/n if n else None for k in ['correct_voxel_fraction','xy_mae_m','same_layer_1m']}))
    atomic_json(dest,dict(kind='offline_native_GT_nearest_LID_quantization_only',records=out,summary=summary,source_sha256=file_hash(__file__),limits='Dense correlated native pixels, not extra independent scenes and not learned feature-ray projection. Exact optical depth vs nearest unchanged 48LID at identical UV. Learned-distribution errors remain separately evaluated on actual feature rays.'))


if __name__=='__main__':main()
