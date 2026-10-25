"""Research-only reference transform and identity-support grading."""
import numpy as np

def valid_rigid(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(T[:3,:3]),1):
        raise ValueError('Proper rigid transform required')
    return T

def camera_in_reference(T_E_R,T_E_C):
    return np.linalg.inv(valid_rigid(T_E_R))@valid_rigid(T_E_C)

def upright_boxes_to_ego(boxes,T_E_R):
    T=valid_rigid(T_E_R)
    if not np.allclose(T[:3,2],[0,0,1],atol=1e-8):
        raise ValueError('Tilted reference requires full orientation representation; cannot silently reduce to yaw')
    a=np.asarray(boxes,float).reshape(-1,9);b=a.copy()
    b[:,:3]=a[:,:3]@T[:3,:3].T+T[:3,3]
    b[:,6]=(a[:,6]+np.arctan2(T[1,0],T[0,0])+np.pi)%(2*np.pi)-np.pi
    b[:,7:9]=np.c_[a[:,7:9],np.zeros(len(a))]@T[:2,:3].T
    return b

def association_grade(views,has_pair):
    if not has_pair:return 'NoSupportedAssociation'
    if any(v['competing_bbox_iou']>=.1 for v in views):return 'AmbiguousAssociation'
    supporting=[v for v in views if v['retained_pixels']>=8 and v['GT_association_purity']>=.9 and v['instance_bbox_iou']>=.5 and v['mask_recall']>=.8]
    return 'IdentitySupported' if len(supporting)>=2 else 'ProjectionProxyOnly'

def pixel_support(mask,bounds):
    vv,uu=np.where(mask)
    if not len(vv):return dict(pixels=0,mask_recall=0.,bbox=None,bbox_iou=0.,inside=0)
    b=np.array([uu.min(),vv.min(),uu.max()+1,vv.max()+1],float)
    if bounds is None:return dict(pixels=len(vv),mask_recall=0.,bbox=b.tolist(),bbox_iou=0.,inside=0)
    from diagnostics.m8b_metrics import iou2
    bounds=np.asarray(bounds);inside=(uu>=bounds[0])&(uu<bounds[2])&(vv>=bounds[1])&(vv<bounds[3])
    return dict(pixels=len(vv),mask_recall=float(inside.mean()),bbox=b.tolist(),bbox_iou=iou2(b,bounds),inside=int(inside.sum()))
