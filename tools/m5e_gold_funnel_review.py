"""Reconcile exact old 663-ray funnel and trace the six surviving rays to coarse Gold."""
import json
from pathlib import Path
import numpy as np
from datasets.m5b_pairs import camera_rays
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5e_gold_funnel_review.json')
    if out.exists():raise FileExistsError(out)
    audit=json.loads(Path('reports/m5e_witness_attrition.json').read_text());root=Path('/root/gpufree-data/hs-drive-data/processed/m5d_index');rows=json.loads((root/'index.json').read_text())['samples'];pairs=json.loads((root/'m5d_pairs.json').read_text())['pairs'];counts={k:0 for k in ['visible','stable_inrange','behind','old_local','semantic_safe','old_reject_planar','old_reject_identity_edge','old_reject_nonplanar']};survivors=[];dense={};g=VoxelGrid()
    for r in audit['records']:
        with np.load(r['path']) as z:d={k:z[k] for k in z.files}
        b=d['rejection_bits'];m=(d['distance']>=20)&(d['distance']<35);counts['visible']+=int(m.sum());m &= (b&3)==0;counts['stable_inrange']+=int(m.sum());m &= (b&4)==0;counts['behind']+=int(m.sum());rej=m&((b&8)>0)
        counts['old_reject_identity_edge']+=int((rej&((b&32)>0)).sum());counts['old_reject_planar']+=int((rej&((b&32)==0)&(d['plane_residual_m']<=.05)).sum());counts['old_reject_nonplanar']+=int((rej&((b&32)==0)&(d['plane_residual_m']>.05)).sum());m &= (b&8)==0;counts['old_local']+=int(m.sum());m &= (b&16)==0;counts['semantic_safe']+=int(m.sum())
        if not m.any():continue
        group=r['group'];empty=next(x for x in rows if x['group']==group and x['configuration']==0)
        if group not in dense:
            raw=Path(empty['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());f=json.loads(Path(empty['raw_frame']).read_text());obs=np.zeros(np.prod(g.shape),bool)
            for cam in f['cameras']:
                rr,_=camera_rays(raw,man,f,cam);ids,inside=g.locate(rr['endpoints']);yes=inside&rr['reliable']&np.isin(rr['semantics'],[2,3]);obs[ids[yes]]=True
            dense[group]=obs
        pair=next(p for p in pairs if p['group']==group and p['a_configuration']==0 and p['b_configuration']==r['configuration'])
        with np.load(pair['path']) as z:
            for i in np.flatnonzero(m):
                rank=int(d['coarse_rank'][i]);survivors.append(dict(group=group,configuration=r['configuration'],camera=r['camera'],uv=d['uv'][i].tolist(),point=d['points_hs'][i].tolist(),rank=rank,grid=list(map(int,np.unravel_index(rank,g.shape))),distance=float(d['distance'][i]),counterfactual_instance=int(d['background_instance_patch'][i,2,2]),coarse_votes=int(z['witness_count_b'].ravel()[rank]),opposite_dense_obstacle=bool(dense[group][rank]),legacy_appeared=bool(z['legacy_appeared'].ravel()[rank]),gold_appeared=bool(z['appeared'].ravel()[rank]),mixed_present=bool(d['mixed_present'][i]),mixed_absent=bool(d['mixed_absent'][i])))
    atomic_json(out,dict(kind='offline_exact_old_validation_funnel_and_coarse_vote_review',ordered_counts=counts,six_ray_survivors=survivors,source_sha256=file_hash(__file__),limits='Reconciles earlier standalone local rejection totals with ordered638->6 funnel. Full gold requires denseveto, >=3votes, legacycell certification. Raypoint absence never proves entirevoxelabsence. Identity edge not automatic proof dangerous occlusion; nonplanarity not uniquelyrendernoise.'))

if __name__=='__main__':main()
