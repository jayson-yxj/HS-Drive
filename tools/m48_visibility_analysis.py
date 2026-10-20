"""Explain collision-query omissions by render visibility, without label tuning."""
import json
from pathlib import Path
import numpy as np
from geometry.reference import transform_points
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m48_visibility_analysis.json')
    if dest.exists():raise FileExistsError(dest)
    records=[];g=VoxelGrid()
    for root in Path('/root/gpufree-data/hs-drive-data/raw').glob('m4[78]_*'):
        for scene in root.glob('*/manifest.json'):
            man=json.loads(scene.read_text());directory=scene.parent
            for fid in man.get('frames',[]):
                files=list(directory.glob(f'*geometry_{fid:08d}.json'))
                if not files:continue
                query=json.loads(files[0].read_text());cal=json.loads((directory/'calibration.json').read_text())['cameras']
                frame=json.loads((directory/'frames'/f'{fid:08d}.json').read_text())
                prefix='m48_heldout_labels' if root.name.startswith('m48_') else 'm48_evidence'
                label=Path('/root/gpufree-data/hs-drive-data/processed')/prefix/man['scene']/f'{fid:08d}.npz'
                if not label.exists():continue
                with np.load(label) as z:surface=z['surface_evidence'];ego=z['ego_mask']
                cameras={}
                for name,c in cal.items():
                    with np.load(directory/frame['cameras'][name]['depth']) as z:d=z['depth_m']
                    cameras[name]=(np.linalg.inv(c['T_hs_ego_from_opt']),np.array(c['K']),d)
                bins={k:dict(n=0,recalled=0) for k in ['render_aligned_015m','render_aligned_050m','not_aligned_any_camera']}
                for q in query['ray_queries']+query['vertical_queries']:
                    for h in q['hits']:
                        sem=man['semantic_groups'].get(str(h['semantic']),0)
                        if sem not in [2,3]:continue
                        point=np.array(h['point_hs']).reshape(3);rank,inside=g.locate(point)
                        if not inside or ego.flat[rank]:continue
                        residuals=[]
                        for inv,k,d in cameras.values():
                            p=transform_points(point,inv).reshape(3)
                            if p[2]<=0:continue
                            uv=k@p;u,v=np.rint(uv[:2]/uv[2]).astype(int)
                            if 0<=u<640 and 0<=v<384:residuals.append(abs(float(d[v,u])-p[2]))
                        closest=min(residuals,default=float('inf'))
                        key='render_aligned_015m' if closest<=.15 else 'render_aligned_050m' if closest<=.5 else 'not_aligned_any_camera'
                        bins[key]['n']+=1;bins[key]['recalled']+=int(surface.flat[rank])
                for b in bins.values():b['recall']=b['recalled']/b['n'] if b['n'] else None
                records.append(dict(scene=man['scene'],frame_id=fid,bins=bins,source_query_sha256=file_hash(files[0])))
    atomic_json(dest,dict(kind='offline_cross_representation_visibility_analysis',records=records,source_sha256=file_hash(__file__),limitations=['Render-depth alignment is diagnostic, not independent truth; <=.15m is fixed reporting tolerance, not used to select training masks.','Unaligned includes occlusion, collision/render geometry discrepancy, rasterization and outside-camera regions. No single-cause attribution.']))
if __name__=='__main__':main()
