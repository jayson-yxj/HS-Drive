"""Observed surface/free occupancy supervision. Independent of learned inputs."""
import numpy as np
from geometry.ray_tracing import VoxelGrid,trace_segment,box_interval
from geometry.reference import unproject_pixels_to_optical,transform_points

SCHEMA='hsd-m45-observed-occupancy-v1'
UNKNOWN,FREE,SURFACE_OCCUPIED=0,1,2
SEM_UNKNOWN,GROUND,STATIC_OBSTACLE,DYNAMIC_OBSTACLE,OTHER=0,1,2,3,4


def sampled_rays(depth,K,cam_to_ego,*,stride=16,min_depth=1.5,max_depth=80.,edge_spread=.5,semantic=None,semantic_map=None):
    v,u=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride];d=depth[v,u]
    patch=np.stack([depth[np.clip(v+dy,0,depth.shape[0]-1),np.clip(u+dx,0,depth.shape[1]-1)] for dy in (-1,0,1) for dx in (-1,0,1)])
    valid=np.isfinite(patch).all(0)&np.isfinite(d)&(d>=min_depth)&(d<max_depth)&((patch.max(0)-patch.min(0))<=edge_spread)
    optical=unproject_pixels_to_optical(u[valid],v[valid],d[valid],K,'optical_z');end=transform_points(optical,cam_to_ego)
    origins=np.broadcast_to(np.asarray(cam_to_ego)[:3,3],end.shape).copy();sem=np.zeros(len(end),np.uint8)
    if semantic is not None:
        if semantic.shape!=depth.shape:raise ValueError('Semantic/depth shape mismatch')
        ids=semantic[v[valid],u[valid]]
        for label,code in (semantic_map or {}).items():sem[ids==int(label)]=code
    return origins,end,sem,{'sampled':int(d.size),'valid_depth_rays':int(valid.sum()),'invalid_depth_rays':int((~valid).sum())}


def build_observed(origins,endpoints,grid=None,*,reliable=None,semantics=None,ego_box=None,surface_margin_m=.1):
    grid=grid or VoxelGrid();o=np.asarray(origins,float);e=np.asarray(endpoints,float)
    if o.shape!=e.shape or o.ndim!=2 or o.shape[1]!=3:raise ValueError('Rays must be Nx3')
    reliable=np.ones(len(e),bool) if reliable is None else np.asarray(reliable,bool)
    semantics=np.zeros(len(e),np.uint8) if semantics is None else np.asarray(semantics,np.uint8)
    if reliable.shape!=(len(e),) or semantics.shape!=(len(e),) or not np.isin(semantics,[0,1,2,3,4]).all():raise ValueError('Ray reliability/semantic schema mismatch')
    if surface_margin_m<0:raise ValueError('Negative margin')
    cells=np.prod(grid.shape);state=np.zeros(cells,np.uint8);hits=np.zeros(cells,np.uint32);vis=np.zeros(cells,np.uint32);conflict=np.zeros(cells,bool);sem_bits=np.zeros(cells,np.uint8)
    accepted=[];self_blocked=0
    for i,(origin,end) in enumerate(zip(o,e)):
        if not reliable[i] or not np.isfinite([origin,end]).all() or np.linalg.norm(end-origin)<1e-8:continue
        if ego_box is not None:
            interval=box_interval(origin,end-origin,*ego_box)
            if interval is not None and min(interval[1],1)>max(interval[0],0)+1e-6:self_blocked+=1;continue
        accepted.append(i);rank,inside=grid.locate(end)
        if inside:
            hits[rank]+=1;state[rank]=SURFACE_OCCUPIED
            if semantics[i]>0:sem_bits[rank]|=np.uint8(1<<int(semantics[i]))
    # Occupied precedence. Any ray whose free segment enters another known surface
    # is truncated there and flagged, including lower-trust free evidence.
    for i in accepted:
        ranks=trace_segment(o[i],e[i],grid,end_margin=surface_margin_m)
        hit_rank,hit_inside=grid.locate(e[i])
        for rank in ranks:
            if hit_inside and rank==hit_rank:break
            if state[rank]==SURFACE_OCCUPIED:conflict[rank]=True;break
            state[rank]=FREE;vis[rank]+=1
        if hit_inside:vis[hit_rank]+=1
    sem=np.zeros(cells,np.uint8);semantic_conflict=np.zeros(cells,bool)
    for code in (GROUND,STATIC_OBSTACLE,DYNAMIC_OBSTACLE,OTHER):
        sem[sem_bits==(1<<code)]=code
    semantic_conflict=(sem_bits!=0)&(sem==0)
    sem_valid=(sem>0)&(state==SURFACE_OCCUPIED)
    obstacle=(sem==STATIC_OBSTACLE)|(sem==DYNAMIC_OBSTACLE);obstacle_valid=sem_valid&(sem!=OTHER)
    # Coarse cells intersecting ego are Unknown, never free/negative supervision.
    ego_mask=np.zeros(cells,bool)
    if ego_box is not None:
        z,x,y=np.indices(grid.shape);edges=grid.edges
        lower=np.stack((edges[0][x],edges[1][y],edges[2][z]),-1)
        upper=np.stack((edges[0][x+1],edges[1][y+1],edges[2][z+1]),-1)
        ego_mask=(np.all(upper>ego_box[0],-1)&np.all(lower<ego_box[1],-1)).ravel()
        state[ego_mask]=UNKNOWN;sem[ego_mask]=0;sem_valid[ego_mask]=False;obstacle[ego_mask]=False;obstacle_valid[ego_mask]=False
    values=dict(occupancy_state=state,occupancy_valid=state!=UNKNOWN,surface_semantic=sem,surface_semantic_valid=sem_valid,
        obstacle_mask=obstacle,obstacle_valid=obstacle_valid,visibility_count=vis,surface_count=hits,ray_conflict_mask=conflict,semantic_conflict_mask=semantic_conflict,ego_mask=ego_mask)
    values={k:v.reshape(grid.shape) for k,v in values.items()};validate_occupancy(values)
    stats={'rays':len(e),'accepted_rays':len(accepted),'self_blocked_rays':self_blocked,'free_voxels':int((state==FREE).sum()),'surface_voxels':int((state==SURFACE_OCCUPIED).sum()),'unknown_voxels':int((state==UNKNOWN).sum()),'ray_conflicts':int(conflict.sum()),'semantic_valid_voxels':int(sem_valid.sum()),'semantic_conflicts':int(semantic_conflict.sum())}
    return values,stats


def validate_occupancy(d):
    s=d['occupancy_state'];shape=s.shape
    if len(shape)!=3 or s.dtype!=np.uint8 or not np.isin(s,[0,1,2]).all():raise ValueError('Invalid occupancy states')
    if not np.array_equal(d['occupancy_valid'],s!=UNKNOWN):raise ValueError('Validity mismatch')
    if any(v.shape!=shape for v in d.values()):raise ValueError('Shape mismatch')
    if np.any(d['surface_semantic_valid']&(s!=SURFACE_OCCUPIED)):raise ValueError('Semantic outside surface')
    if np.any(d['obstacle_mask']&~d['obstacle_valid']):raise ValueError('Unknown obstacle marked positive')
