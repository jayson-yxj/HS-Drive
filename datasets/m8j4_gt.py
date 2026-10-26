"""Supervision-only geometry and conservative render-region association.

Validity describes evidence, never an exhaustive negative/background mask.
"""
import cv2
import numpy as np
from geometry.reference import CARLA_EGO_TO_HS_EGO as S, OPTICAL_TO_CARLA_CAM as O


def transform(points, matrix):
    p = np.asarray(points, float).reshape(-1, 3)
    return (np.c_[p, np.ones(len(p))] @ np.asarray(matrix).T)[:, :3]


def box_to_hs(obj, world2ego):
    result = dict(actor_id=str(obj.get('id')), source_class=obj.get('class'),
                  source_type=obj.get('type_id'), geometry_status='Unknown',
                  training_negative=False, reference='ActorBoxCenter_NotMeshContact')
    if obj.get('class') not in ('vehicle', 'walker'):
        result['reason'] = 'UnsupportedClassOrNoEightCornerReference'
        return result
    try:
        center = np.asarray(obj['center'], float)
        extent = np.asarray(obj['extent'], float)
        corners = np.asarray(obj['world_cord'], float)
        assert center.shape == extent.shape == (3,) and corners.shape == (8,3)
        assert np.isfinite(np.r_[center,extent,corners.ravel()]).all() and (extent>0).all()
        T = S @ np.asarray(world2ego)
        cs = transform(corners,T); c=transform(center,T)[0]
        axes = np.stack([cs[0]-cs[4],cs[0]-cs[2],cs[0]-cs[1]],1)
        dims = np.linalg.norm(axes,axis=0); basis=axes/dims
        error = float(np.max(abs(dims-2*extent)))
        orth = float(np.max(abs(basis.T @ basis-np.eye(3))))
        assert error<.01 and orth<.01 and np.linalg.norm(c-cs.mean(0))<.005
        yaw=float(np.arctan2(basis[1,0],basis[0,0]))
        tilt=float(np.rad2deg(np.arccos(np.clip(abs(basis[2,2]),0,1))))
        cls='pedestrian' if obj['class']=='walker' else ('car' if obj.get('base_type')=='car' else 'UnknownVehicleSubtype')
        result.update(center_hs_m=c.tolist(),dimensions_lwh_m=dims.tolist(),yaw_ccw_rad=yaw,
            corners_hs_m=cs.tolist(),basis_hs=basis.tolist(),tilt_deg=tilt,
            dimension_residual_m=error,orthogonality_residual=orth,model_class=cls,
            geometry_status='Valid',yaw_only_comparable=tilt<=2,
            distance_xy_m=float(np.linalg.norm(c[:2])))
    except (KeyError,ValueError,AssertionError,TypeError):
        result.update(geometry_status='Invalid',reason='InvalidBoxFieldsOrGeometry')
    return result


def project_box(corners, world2cam, K, width, height, near=.1):
    p=transform(corners,np.linalg.inv(O) @ world2cam)
    if not np.isfinite(p).all():
        return dict(status='Invalid',reason='NonfiniteProjection')
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
    points=[q for q in p if q[2]>=near]
    for i,j in edges:
        a,b=p[i],p[j]
        if (a[2]<near)!=(b[2]<near):
            points.append(a+(b-a)*((near-a[2])/(b[2]-a[2])))
    if not points:
        return dict(status='Invalid',reason='BehindCamera',visible_in_this_view=False)
    q=np.asarray(points); uv=(np.asarray(K)@q.T).T;uv=uv[:,:2]/uv[:,2:]
    hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
    viewport=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    area, clipped=cv2.intersectConvexConvex(hull,viewport)
    if clipped is None or area<1:
        return dict(status='Invalid',reason='OutsideImageOrSubpixel',visible_in_this_view=False)
    poly=clipped.reshape(-1,2)
    low=poly.min(0);high=poly.max(0)
    trunc=bool((uv[:,0]<0).any() or (uv[:,0]>=width).any() or (uv[:,1]<0).any() or (uv[:,1]>=height).any())
    return dict(status='Valid',polygon=poly.tolist(),bbox_xyxy=[*low.tolist(),*high.tolist()],
                unclipped_bbox=[*uv.min(0).tolist(),*uv.max(0).tolist()],
                polygon_area_px=float(area),truncated=trunc,near_plane_clipped=bool((p[:,2]<near).any()),
                optical_z_range=[float(p[:,2].min()),float(p[:,2].max())],
                visible_in_this_view='UnknownUntilRenderEvidence')


def regions(instance, semantic):
    """Treat (semantic R, G low-byte, B high-byte) as a render token, not Actor ID."""
    if instance.ndim!=3 or instance.shape[2]!=4 or instance.dtype!=np.uint8:
        raise ValueError('Expected raw BGRA instance PNG')
    labels=instance[:,:,2]
    tokens=instance[:,:,1].astype(np.int32)+256*instance[:,:,0].astype(np.int32)
    out=[]
    for sem in (12,14,15,16,18,19):
        ys,xs=np.nonzero(labels==sem)
        if not len(xs):continue
        ids,inverse,counts=np.unique(tokens[ys,xs],return_inverse=True,return_counts=True)
        minx=np.full(len(ids),instance.shape[1]);miny=np.full(len(ids),instance.shape[0])
        maxx=np.zeros(len(ids),int);maxy=np.zeros(len(ids),int)
        np.minimum.at(minx,inverse,xs);np.minimum.at(miny,inverse,ys)
        np.maximum.at(maxx,inverse,xs);np.maximum.at(maxy,inverse,ys)
        agree=np.bincount(inverse,weights=(semantic[ys,xs]==sem),minlength=len(ids))
        for i,key in enumerate(ids):
            out.append(dict(render_token=int(key),semantic=int(sem),pixels=int(counts[i]),
                bbox_xyxy=[int(minx[i]),int(miny[i]),int(maxx[i]+1),int(maxy[i]+1)],
                semantic_agreement=float(agree[i]/counts[i])))
    return labels,tokens,out


def bbox_iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def associate(projection,source_class,labels,tokens,candidates,cfg):
    if projection['status']!='Valid':
        return dict(status='Invalid',reason=projection['reason'],match=None)
    if projection['near_plane_clipped']:
        return dict(status='Unknown',reason='NearPlaneIntersectionRequiresReview',match=None)
    x0,y0,x1,y1=projection['bbox_xyxy']
    x0=max(0,int(np.floor(x0)));y0=max(0,int(np.floor(y0)))
    x1=min(labels.shape[1],int(np.ceil(x1))+1);y1=min(labels.shape[0],int(np.ceil(y1))+1)
    mask=np.zeros((y1-y0,x1-x0),np.uint8)
    polygon=np.asarray(projection['polygon'])-[x0,y0]
    cv2.fillConvexPoly(mask,np.round(polygon).astype(np.int32),1)
    target_sem={12} if source_class=='walker' else {14,15,16,18,19}
    scored=[]
    for r in candidates:
        if r['semantic'] not in target_sem or r['pixels']<cfg['min_region_pixels']:continue
        if bbox_iou(r['bbox_xyxy'],projection['bbox_xyxy'])<=0:continue
        overlap=int(np.count_nonzero(mask & (labels[y0:y1,x0:x1]==r['semantic']) &
                                     (tokens[y0:y1,x0:x1]==r['render_token'])))
        if overlap<cfg['min_region_pixels']:continue
        containment=overlap/r['pixels'];iou=bbox_iou(r['bbox_xyxy'],projection['bbox_xyxy'])
        item=dict(**r,overlap_pixels=overlap,containment=containment,bbox_iou=iou,
                  silhouette_fraction=overlap/max(1,int(mask.sum())),
                  score=.5*containment+.5*iou)
        item['qualified']=(containment>=cfg['min_region_containment'] and iou>=cfg['min_bbox_iou']
                           and r['semantic_agreement']>=cfg['min_semantic_agreement'])
        scored.append(item)
    scored.sort(key=lambda r:r['score'],reverse=True)
    if not scored:
        return dict(status='Unknown',reason='NoCompatibleVisibleRegion_OccludedOrMissingUnknown',match=None)
    best=scored[0];margin=best['score']-(scored[1]['score'] if len(scored)>1 else 0)
    supported=best['qualified'] and margin>=cfg['min_score_margin']
    return dict(status='Valid' if supported else 'Unknown',
                reason='EvidenceSupportedCandidate' if supported else 'AmbiguousOrWeakEvidence',
                match=best if supported else None,candidates=scored[:3],score_margin=margin,
                occlusion_state='PartialOcclusionProxy_UnknownTrueFraction' if best['silhouette_fraction']<cfg['occlusion_proxy_fraction'] else 'VisibleSupported_NoFullVisibilityClaim',
                human_review_required=True)


def enforce_one_to_one(views):
    owners={}
    for actor_id,v in views.items():
        m=v.get('match')
        if v['status']=='Valid':
            owners.setdefault((m['semantic'],m['render_token']),[]).append(actor_id)
    for key,ids in owners.items():
        if len(ids)>1:
            for aid in ids:
                views[aid].update(status='Unknown',reason='SharedRenderRegion_ManyToOne',
                                  rejected_match=views[aid]['match'],match=None)
    return views
