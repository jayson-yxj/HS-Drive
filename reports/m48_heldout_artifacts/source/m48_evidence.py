"""Spatial Evidence V2: nonexclusive observation events, NEVER traversability.

Negatives mean absence in this fixed, finite measurement recipe on an observed
cell, not proof of empty volume. Only supplied visibility-certified rays count.
"""
from dataclasses import asdict, dataclass
import numpy as np
from geometry.ray_tracing import VoxelGrid, trace_segment

SCHEMA = 'hsd-m48-spatial-evidence-v2'

@dataclass(frozen=True)
class EvidenceConfig:
    fine_height_m: float = 0.25
    surface_margin_m: float = 0.1
    free_stride: int = 16
    surface_stride: int = 1
    eval_modulus: int = 4
    eval_remainder: int = 1


def build_evidence(origins, endpoints, surface_points, semantics, *, ego_box=None,
                   grid=None, config=EvidenceConfig(), reliable=None):
    """Fine-Z surface priority only within a fine cell; coarse OR preserves mix.

    XY unchanged; each 1.5m coarse layer contains six .25m sublayers. Fine free
    paths terminate at known fine surface cells, including another ray's hit.
    No occupancy dilation, whole-cell-free target or implicit unknown negative.
    """
    g = grid or VoxelGrid()
    o, e, pts = [np.asarray(v, dtype=float).reshape(-1, 3) for v in (origins, endpoints, surface_points)]
    sem = np.asarray(semantics, dtype=np.uint8)
    if o.shape != e.shape or sem.shape != (len(pts),) or not np.isin(sem, [0,1,2,3,4]).all():
        raise ValueError('Invalid ray/semantic shapes')
    if config.surface_margin_m < 0 or config.fine_height_m <= 0:
        raise ValueError('Invalid evidence configuration')
    z = []
    for low, high in zip(g.z[:-1], g.z[1:]):
        count = round((high-low)/config.fine_height_m)
        if count < 1 or not np.isclose(count*config.fine_height_m, high-low):
            raise ValueError('Fine height must divide each coarse layer')
        z.extend(np.linspace(low, high, count+1)[:-1])
    fine = VoxelGrid(g.x, g.y, tuple(z+[g.z[-1]]), g.resolution)
    nf, nc = int(np.prod(fine.shape)), int(np.prod(g.shape))
    fine_z = np.searchsorted(g.z, (np.array(fine.z[:-1])+fine.z[1:])/2, side='right')-1
    mapping = (fine_z[:,None]*np.prod(g.shape[1:])+np.arange(np.prod(g.shape[1:]))).ravel()
    counts = np.zeros(nf, np.uint32); bits = np.zeros(nf, np.uint8)
    ids, inside = fine.locate(pts)
    np.add.at(counts, ids[inside], 1)
    for code, bit in [(1,1),(2,2),(3,2),(4,4),(0,8)]:
        np.bitwise_or.at(bits, ids[inside & (sem==code)], np.uint8(bit))
    surface = counts > 0
    free = np.zeros(nf, bool); conflict = np.zeros(nf, bool)
    reliable = np.ones(len(o),bool) if reliable is None else np.asarray(reliable,bool)
    if reliable.shape != (len(o),): raise ValueError('Invalid reliable mask')
    accepted = 0
    for origin, end, ok in zip(o,e,reliable):
        if not ok or not np.isfinite([origin,end]).all(): continue
        accepted += 1
        for rank in trace_segment(origin,end,fine,end_margin=config.surface_margin_m):
            if surface[rank]:
                conflict[rank] = True
                break
            free[rank] = True
    def aggregate(mask):
        out=np.zeros(nc,bool);np.logical_or.at(out,mapping,mask);return out.reshape(g.shape)
    s, f, uncertain = aggregate(surface), aggregate(free), aggregate(conflict)
    coarse_bits=np.zeros(nc,np.uint8);np.bitwise_or.at(coarse_bits,mapping,bits)
    coarse_bits=coarse_bits.reshape(g.shape)
    semantic=np.stack([(coarse_bits & bit)>0 for bit in (1,2,4)])
    # Ground and obstacle in one *fine* cell: unresolved semantic geometry.
    semantic_conflict=aggregate(((bits&1)>0)&((bits&2)>0))
    ego=np.zeros(g.shape,bool)
    if ego_box is not None:
        iz,ix,iy=np.indices(g.shape);edges=g.edges
        lower=np.stack((edges[0][ix],edges[1][iy],edges[2][iz]),-1)
        upper=np.stack((edges[0][ix+1],edges[1][iy+1],edges[2][iz+1]),-1)
        ego=np.all(upper>ego_box[0],-1)&np.all(lower<ego_box[1],-1)
    s &= ~ego;f &= ~ego;uncertain &= ~ego
    obs=s|f
    sv=obs.copy();fv=obs & ~uncertain
    semantic_valid=s & ((coarse_bits&8)==0) & ~semantic_conflict
    semantic &= s[None]
    obstacle_valid=semantic_valid & (semantic[0]|semantic[1]) & ~semantic[2]
    d=dict(surface_evidence=s, free_ray_evidence=f, surface_valid=sv,
        free_evidence_valid=fv, observability=obs,
        mixed_cell_mask=((s&f)|uncertain|semantic_conflict)&~ego,
        surface_semantic=semantic, surface_semantic_valid=semantic_valid,
        obstacle_valid=obstacle_valid, evidence_conflict_mask=uncertain,
        semantic_conflict_mask=semantic_conflict&~ego, ego_mask=ego,
        fine_surface_fraction=np.bincount(mapping,weights=surface,minlength=nc).reshape(g.shape)/np.bincount(mapping,minlength=nc).reshape(g.shape),
        fine_free_ray_fraction=np.bincount(mapping,weights=free,minlength=nc).reshape(g.shape)/np.bincount(mapping,minlength=nc).reshape(g.shape))
    validate_evidence(d,g.shape)
    return d,dict(schema=SCHEMA,config=asdict(config),accepted_free_rays=accepted,
        surface_cells=int(s.sum()),free_evidence_cells=int(f.sum()),
        mixed_cells=int(d['mixed_cell_mask'].sum()),fine_conflict_cells=int(conflict.sum()),
        valid_coverage=float(obs.mean()),unknown_fraction=float((~obs).mean()),
        mixed_cell_ratio=float(d['mixed_cell_mask'].sum()/max(1,obs.sum())),
        evidence_conflict_rate=float(uncertain.sum()/max(1,obs.sum())),
        free_loss_mask_coverage=float(fv.mean()),surface_loss_mask_coverage=float(sv.mean()))


def validate_evidence(d, shape=(4,160,120)):
    for key, value in d.items():
        expected=(3,*shape) if key=='surface_semantic' else shape
        if value.shape != expected or not np.isfinite(value).all(): raise ValueError('Evidence shape/finite')
    obs=d['observability']
    if not np.array_equal(obs,d['surface_evidence']|d['free_ray_evidence']):raise ValueError('Observability mismatch')
    for name in ('surface_valid','free_evidence_valid','surface_semantic_valid','obstacle_valid'):
        if (d[name]&~obs).any() or (d[name]&d['ego_mask']).any():raise ValueError('Unknown/ego supervision')
    if (d['surface_semantic_valid']&~d['surface_evidence']).any():raise ValueError('Semantic outside surface')
    if (d['free_evidence_valid']&d['evidence_conflict_mask']).any():raise ValueError('Unresolved crossing supervised')
