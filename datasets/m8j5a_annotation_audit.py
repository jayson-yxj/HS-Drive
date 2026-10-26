"""Lossless supervision geometry, independent of visual identity and training.

Existing J4 conversion is retained as a measured baseline, not overwritten.
Unverified identity never deletes a geometrically valid source object.
"""
import numpy as np

from datasets.m8j4_gt import project_box,transform
from geometry.reference import CARLA_EGO_TO_HS_EGO as S


SIGNS=np.array([[x,y,z] for x in (1,-1) for y in (1,-1) for z in (1,-1)],float)


def raw_corners(obj):
    """Prefer supplied world vertices. Level traffic boxes follow official viz.

    For missing vehicle vertices use bbx_loc plus inverse world2vehicle, never
    unreliable static Actor location/rotation. Missing local center stays unknown.
    """
    if 'world_cord' in obj:
        p=np.asarray(obj['world_cord'],float)
        if p.shape!=(8,3) or not np.isfinite(p).all():raise ValueError('MalformedWorldCorners')
        return p,'RecordedWorldCorners'
    extent=np.asarray(obj.get('extent'),float)
    if extent.shape!=(3,) or not np.isfinite(extent).all() or (extent<=0).any():
        raise ValueError('MissingOrInvalidExtent')
    if obj['class'] in ('vehicle','walker','ego_vehicle'):
        matrix_key={'vehicle':'world2vehicle','walker':'world2ped','ego_vehicle':'world2ego'}[obj['class']]
        if 'bbx_loc' not in obj or matrix_key not in obj:raise ValueError('NoReliableActorLocalReference')
        local=np.asarray(obj['bbx_loc'],float)
        return transform(SIGNS*extent+local,np.linalg.inv(np.asarray(obj[matrix_key]))),'ActorLocalBoxInverseWorldTransform'
    if obj['class'] in ('traffic_light','traffic_sign') and 'center' in obj:
        # Official visualize.py uses calculate_cube_vertices(center,extent).
        # This does not certify orientation or physical mesh/trigger geometry.
        return SIGNS*extent+np.asarray(obj['center']),'OfficialLevelBoxVisualizationProxy_NotDetectionGT'
    raise ValueError('UnsupportedSourceGeometry')


def converted_geometry(obj,world2ego):
    result=dict(actor_id=str(obj['id']),source_class=obj['class'],state=obj.get('state'),
        geometry_status='Unknown',identity_status='NotRequiredForCoordinateConversion',
        raw_record_preserved=True,training_authorized=False,background_negative=False)
    try:
        world,basis=raw_corners(obj);hs=transform(world,S@np.asarray(world2ego))
        axes=np.stack([hs[0]-hs[4],hs[0]-hs[2],hs[0]-hs[1]],1)
        dims=np.linalg.norm(axes,axis=0)
        if (dims<=0).any():raise ValueError('DegenerateCorners')
        if obj['class'] in ('vehicle','walker','ego_vehicle'):
            if np.max(abs(world.mean(0)-np.asarray(obj['center'])))>.01:
                raise ValueError('SourceCenterCornerDisagreement')
            if np.max(abs(dims-2*np.asarray(obj['extent'])))>.01:
                raise ValueError('SourceExtentCornerDisagreement')
        result.update(geometry_status='Valid',reference=basis,corners_hs_m=hs.tolist(),
            center_hs_m=transform(np.asarray(obj['center'])[None],S@np.asarray(world2ego))[0].tolist(),
            dimensions_lwh_m=dims.tolist(),yaw_ccw_rad=float(np.arctan2(axes[1,0],axes[0,0])),
            source_class_mapping={'vehicle':'vehicle','walker':'pedestrian'}.get(obj['class'],'NotPETRDetectionClass'))
    except (ValueError,KeyError,TypeError,np.linalg.LinAlgError) as exc:
        result['reason']=str(exc)
    return result


def source_geometry_residuals(obj):
    """Three independent source checks, without assuming center is actor origin."""
    world,basis=raw_corners(obj)
    if obj['class'] not in ('vehicle','walker','ego_vehicle'):return dict(reference=basis)
    key={'vehicle':'world2vehicle','walker':'world2ped','ego_vehicle':'world2ego'}[obj['class']]
    local=np.asarray(obj['bbx_loc'],float);extent=np.asarray(obj['extent'],float)
    rebuilt=transform(SIGNS*extent+local,np.linalg.inv(np.asarray(obj[key])))
    center=transform(local[None],np.linalg.inv(np.asarray(obj[key])))[0]
    dims=np.linalg.norm(np.stack([world[0]-world[4],world[0]-world[2],world[0]-world[1]],1),axis=0)
    return dict(reference=basis,reconstructed_corners_max_m=float(abs(rebuilt-world).max()),
        reconstructed_center_max_m=float(abs(center-np.asarray(obj['center'])).max()),
        centroid_center_max_m=float(abs(world.mean(0)-obj['center']).max()),
        extent_halfsize_max_m=float(abs(dims-2*extent).max()))


def project_both(obj,converted,sensor,world2ego):
    world,_=raw_corners(obj)
    direct=project_box(world,np.asarray(sensor['world2cam']),np.asarray(sensor['intrinsic']),1600,900)
    cam_from_hs=np.asarray(sensor['world2cam'])@np.linalg.inv(np.asarray(world2ego))@S
    via_hs=project_box(converted['corners_hs_m'],cam_from_hs,np.asarray(sensor['intrinsic']),1600,900)
    if direct['status']!=via_hs['status']:raise ValueError('ProjectionStatusMismatch')
    residual=None
    if direct['status']=='Valid':residual=float(abs(np.asarray(direct['bbox_xyxy'])-via_hs['bbox_xyxy']).max())
    return direct,via_hs,residual
