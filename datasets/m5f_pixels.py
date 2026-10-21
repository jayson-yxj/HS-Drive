"""Versioned visible first-surface pixel target. Negative is NOT 3D free."""
import cv2
import numpy as np
from geometry.reference import unproject_pixels_to_optical,transform_points

SCHEMA='hsd-m5f-visible-first-surface-v1'
VEHICLE=(14,15,16,17,18,19)
PEDESTRIAN=(12,13)
NEGATIVE=(1,2,3,4,5,6,7,8,9,10,24,25,26,27,28)


def homogeneous(semantic,instance):
    result=np.ones(semantic.shape,bool)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            result &= semantic==np.roll(semantic,(dy,dx),(0,1))
            result &= instance==np.roll(instance,(dy,dx),(0,1))
    result[[0,-1]]=False;result[:,[0,-1]]=False
    return result


def adapt(depth,semantic,instance,self_mask,K,T,actors):
    """GT-only adapter. Render identity matched by point containment, never ==actorID.

    Association: nonself target-class points, >=3 endpoint pixels in AABB±.10m,
    >=90% of all eligible pixels for render identity contained in only this box.
    AABB is association proxy, not mesh occupied volume or precise identity proof.
    Dynamic21 only accepted for geometrically associated trafficcone01 identities;
    all other Dynamic/Static/Other/Water/unknown ignored, not non-target negative.
    """
    if any(a.shape!=depth.shape for a in [semantic,instance,self_mask]):raise ValueError('unaligned raster shapes')
    h,w=depth.shape;v,u=np.mgrid[:h,:w]
    finite=np.isfinite(depth)&(depth>=1.5)&(depth<80)
    pts=transform_points(unproject_pixels_to_optical(u,v,np.where(finite,depth,0),K,'optical_z'),T).reshape(h,w,3)
    interior=homogeneous(semantic,instance)
    valid_depth=finite&~self_mask;valid=valid_depth&interior
    target=np.isin(semantic,VEHICLE+PEDESTRIAN)
    known=target|np.isin(semantic,NEGATIVE)
    classes=np.zeros(depth.shape,np.uint8);classes[np.isin(semantic,VEHICLE)]=1;classes[np.isin(semantic,PEDESTRIAN)]=2
    actor_map=np.zeros(depth.shape,np.int16);claims=[];metadata=[]
    for j,a in enumerate(actors):
        box=np.array(a['corners_hs_ego_m']);center=box.mean(0)
        if a['is_ego'] or np.linalg.norm(center[:2])>=80:continue
        typ=a['type_id'];tags=VEHICLE if typ.startswith('vehicle.') else PEDESTRIAN if typ.startswith('walker.') else (21,) if typ=='static.prop.trafficcone01' else ()
        allowed=valid_depth&np.isin(semantic,tags)&(instance!=0)
        inside=((pts>=box.min(0)-.1)&(pts<=box.max(0)+.1)).all(-1)
        ids=[];detail=[]
        for identity in np.unique(instance[allowed&inside]):
            whole=allowed&(instance==identity);support=int((whole&inside).sum());den=int(whole.sum());purity=support/max(den,1)
            accepted=support>=3 and purity>=.9
            detail.append(dict(render_id=int(identity),support=support,total=den,purity=purity,accepted=accepted,equals_actor_id=int(identity)==int(a['actor_id'])))
            if accepted:ids.append(identity)
        mask=allowed&np.isin(instance,ids);claims.append((j+1,typ,mask,inside))
        metadata.append(dict(local_id=j+1,actor_id=a['actor_id'],type_id=typ,distance_m=float(np.linalg.norm(center[:2])),center=center.tolist(),identity_candidates=detail,reference='renderidentity geometrically associated with AABB, not exactMesh identity'))
    overlaps=np.zeros(depth.shape,np.int16)
    for _,_,m,_ in claims:overlaps+=m
    for idx,typ,m,inside in claims:
        m &= overlaps==1
        if typ=='static.prop.trafficcone01':target|=m;known|=m;classes[m]=3
        actor_map[m&valid]=idx
    valid &= known&(overlaps<=1)
    target &= valid;classes[~target]=0;actor_map[~target]=0
    for a in metadata:
        m=actor_map==a['local_id'];a['pixels']=int(m.sum());a['visible_eligible']=a['pixels']>=3
        yy,xx=np.where(m);a['bbox_uv']=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None
    return dict(target=target,valid=valid,classes=classes,actor=actor_map,depth=depth.astype(np.float32),radial=np.linalg.norm(pts[...,:2],axis=-1).astype(np.float32),points=pts.astype(np.float32),interior=interior,self_mask=self_mask),metadata
