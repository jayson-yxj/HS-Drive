"""Frozen paired V1/V2 evaluation on render and supplemental collision surfaces."""
import argparse,json,time
from pathlib import Path
import numpy as np
from datasets.occupancy_quality import visibility_rays,build_visibility_observed,refine_surface_evidence
from datasets.m48_evidence import build_evidence,EvidenceConfig
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import freeze_sources


def read(p):return json.loads(Path(p).read_text())

def frame_rays(root,man,fid):
    cal=read(root/'calibration.json')['cameras'];frame=read(root/'frames'/f'{fid:08d}.json')
    ref=read(root/'frames'/f"{man['counterfactual_frame']:08d}.json")
    box=np.array(next(a for a in frame['actors'] if a['is_ego'])['corners_hs_ego_m'])
    box=(box.min(0)-.05,box.max(0)+.05);parts=[]
    for camera,c in cal.items():
        def arr(f):
            r=f['cameras'][camera]
            for k in ['depth','instance']:assert file_hash(root/r[k])==r[k+'_sha256']
            with np.load(root/r['depth']) as a:d=a['depth_m']
            with np.load(root/r['instance']) as a:s,i=a['semantic'],a['instance']
            return d,s,i
        ray=visibility_rays(*arr(frame),*arr(ref),c['K'],c['T_hs_ego_from_opt'],box,man['semantic_groups'],stride=1)
        parts.append({k:ray[k][ray['reliable']] for k in ['origins','endpoints','semantics','uv']})
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]},box


def metrics(points,sem,v1,v2):
    g=VoxelGrid();r,inside=g.locate(points);r=r[inside];sem=np.asarray(sem)[inside]
    keep=~v2['ego_mask'].ravel()[r];r=r[keep];sem=sem[keep];pts=np.asarray(points)[inside][keep]
    a=v1['occupancy_state'].ravel()[r];b=v2['surface_evidence'].ravel()[r]
    obs=np.isin(sem,[2,3]);ground=sem==1
    def score(mask):
        def mean(vals):return float(np.mean(vals)) if len(vals) else None
        return dict(n=int(mask.sum()),v1_surface_recall=mean(a[mask]==2),v2_surface_recall=mean(b[mask]),
            v1_free_surface_conflict_rate=mean(a[mask]==1),
            v2_surface_with_free_event=mean(v2['free_ray_evidence'].ravel()[r[mask]]&b[mask]),
            v2_surface_missing=mean(~b[mask]))
    output=dict(all=score(np.ones(len(r),bool)),obstacle=score(obs),ground=score(ground))
    dist=np.linalg.norm(pts[:,:2],axis=1);layer=np.unravel_index(r,g.shape)[0]
    output['by_distance']={f'{lo}-{hi}':dict(all=score((dist>=lo)&(dist<hi)),obstacle=score(obs&(dist>=lo)&(dist<hi))) for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,80)]}
    output['by_layer']={str(i):dict(all=score(layer==i),obstacle=score(obs&(layer==i))) for i in range(4)}
    # Nominal straight vehicle corridor only; surface risk diagnostic, not planner free.
    corridor=(pts[:,0]>0)&(pts[:,0]<20)&(abs(pts[:,1])<1.3)&(pts[:,2]>-.5)&(pts[:,2]<2.5)
    output['nominal_footprint_corridor_obstacle']=score(corridor&obs)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',nargs='+',required=True)
    p.add_argument('--output',default='reports/m48_spatial_evidence.json')
    p.add_argument('--labels',default='/root/gpufree-data/hs-drive-data/processed/m48_evidence')
    p.add_argument('--artifacts',default='reports/m48_spatial_artifacts');a=p.parse_args()
    for path in [a.output,a.labels,a.artifacts]:
        if Path(path).exists():raise FileExistsError(path)
    dest=Path(a.labels);dest.mkdir(parents=True);art=Path(a.artifacts);art.mkdir()
    sources=freeze_sources(art/'source',[__file__,'datasets/m48_evidence.py','datasets/occupancy_quality.py','geometry/ray_tracing.py'])
    audit=read('reports/m47_false_free_audit.json')['records'];records=[];start=time.perf_counter()
    for path in a.raw:
        campaign=Path(path)
        for root in sorted(campaign.iterdir()):
            if not (root/'manifest.json').exists():continue
            man=read(root/'manifest.json')
            for fid in man['frames']:
                rays,box=frame_rays(root,man,fid);uv=rays['uv']
                sparse=(uv%16==0).all(-1);dense=(uv%2==0).all(-1);held=(uv[:,0]+2*uv[:,1])%4==1
                sr={k:rays[k][sparse] for k in ['origins','endpoints','semantics']};sr['reliable']=np.ones(sparse.sum(),bool)
                v1,_=build_visibility_observed(sr,box);v1,_=refine_surface_evidence(v1,rays['endpoints'][dense],rays['semantics'][dense])
                matched,ms=build_evidence(sr['origins'],sr['endpoints'],rays['endpoints'][dense],rays['semantics'][dense],ego_box=box)
                v2,stats=build_evidence(sr['origins'],sr['endpoints'],rays['endpoints'][~held],rays['semantics'][~held],ego_box=box)
                folder=dest/man['scene'];folder.mkdir(exist_ok=True);label=folder/f'{fid:08d}.npz';np.savez_compressed(label,**v2)
                collision=[];sem=[]
                paths=list(root.glob(f'*geometry_{fid:08d}.json'))
                if paths:
                    queries=read(paths[0])
                    for q in queries['ray_queries']+queries['vertical_queries']:
                        for hit in q['hits']:
                            collision.append(np.asarray(hit['point_hs']).reshape(3));sem.append(man['semantic_groups'].get(str(hit['semantic']),0))
                rows=[r for r in audit if r['scene']==man['scene'] and r['frame_id']==fid]
                audit_stats=[dict(id=r['id'],v2_surface=bool(v2['surface_evidence'].flat[r['rank']]),v2_free=bool(v2['free_ray_evidence'].flat[r['rank']]),mixed=bool(v2['mixed_cell_mask'].flat[r['rank']])) for r in rows]
                record=dict(scene=man['scene'],frame_id=fid,map=man['map'],vehicle=man.get('vehicle','vehicle.tesla.model3'),
                    role=man.get('role','historical_development'),label=str(label),label_sha256=file_hash(label),stats=stats,
                    v1_valid_coverage=float(v1['occupancy_valid'].mean()),v2_same_sampling=ms,
                    render_heldout=metrics(rays['endpoints'][held],rays['semantics'][held],v1,v2),
                    collision_reference=metrics(np.array(collision).reshape(-1,3),sem,v1,v2) if collision else None,
                    collision_same_sampling=metrics(np.array(collision).reshape(-1,3),sem,v1,matched) if collision else None,
                    audit104=audit_stats,raw_manifest_sha256=file_hash(root/'manifest.json'))
                records.append(record);print(man['scene'],fid,stats['valid_coverage'],flush=True)
    assert all(file_hash(k)==v for k,v in sources.items())
    atomic_json(a.output,dict(kind='offline_paired_evaluation_actual_carla_records',records=records,sources=sources,elapsed_s=time.perf_counter()-start,
        recipe='V1 sparse16+dense2. V2 same-sampling vs V2 native nonheld surface; free16 unchanged. Evaluation pixels (u+2v)%4==1 never generated labels.',
        negative_definition='absence of this finite observation event on observed cells, NOT absence of physical surfaces or whole-cell free',
        limitations=['Render heldout pixels share renderer, not independent 3D GT.','Collision cast_ray is supplemental geometry and includes occluded hits, not exact render mesh.','Coverage is event support, not volume certified empty.'],
        formal_training_approved=False))
if __name__=='__main__':main()
