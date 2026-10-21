"""RGB-only connected components and explicitly conditional flat-ground hypotheses."""
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment

def components(q,threshold=.9,min_pixels=6):
    n,labels,stats,_=cv2.connectedComponentsWithStats((q>=threshold).astype(np.uint8),8)
    result=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<min_pixels or h<3:continue
        mask=labels==i;v,u=np.where(mask);bottom=int(v.max());bu=u[v>=bottom-1];uv=np.array([float(np.median(bu)),float(bottom)])
        result.append(dict(mask=mask,bbox=[int(x),int(y),int(w),int(h)],area=int(area),uv=uv))
    return result

def plane_intersection(uv,K,T,plane_z=-.25):
    direction=T[:3,:3]@np.linalg.solve(K,np.array([*uv,1.]));origin=T[:3,3]
    if direction[2]>=-.02:return None,'horizon_or_upward'
    z=(plane_z-origin[2])/direction[2];point=origin+z*direction
    if z<=0 or np.linalg.norm(point[:2])>50:return None,'range'
    return point,None

def locate(component,K,T,shape=(384,640)):
    x,y,w,h=component['bbox'];uv=component['uv']
    if x<=1 or y<=1 or x+w>=shape[1]-1 or y+h>=shape[0]-1:return None,'image_boundary'
    if w/max(h,1)>4:return None,'wide_or_merged'
    point,reason=plane_intersection(uv,K,T)
    if point is None:return None,reason
    alternatives=[plane_intersection(uv+[0,d],K,T,z)[0]for d in [-1,1]for z in [-.35,-.15]]
    if any(p is None for p in alternatives)or max(np.linalg.norm(p[:2]-point[:2])for p in alternatives)>3:return None,'ground_or_pixel_sensitive'
    return point,'conditional_flat_plane_contact_unverified'

def match(candidates,truth):
    if not candidates or not truth:return [],set(range(len(candidates))),set(range(len(truth)))
    score=np.array([[np.logical_and(c['mask'],t).sum()/max(np.logical_or(c['mask'],t).sum(),1)for t in truth]for c in candidates]);ii,jj=linear_sum_assignment(-score);pairs=[(int(i),int(j),float(score[i,j]))for i,j in zip(ii,jj)if score[i,j]>=.1]
    return pairs,set(range(len(candidates)))-{i for i,_,_ in pairs},set(range(len(truth)))-{j for _,j,_ in pairs}
