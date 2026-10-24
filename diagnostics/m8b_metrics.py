"""Evaluator-only geometry and matching; never imported by the inference worker."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from datasets.m8b_petr_adapter import corners_hs,A

def coarse(label):
    return 'person' if label==8 else 'cone' if label==9 else 'unsupported_barrier' if label==5 else 'vehicle'

def summary(x):
    v=np.asarray(x,float)
    return dict(n=len(v),mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,p90=float(np.percentile(v,90)) if len(v) else None,p95=float(np.percentile(v,95)) if len(v) else None,max=float(v.max()) if len(v) else None,over5_rate=float((v>5).mean()) if len(v) else None)

def project_bounds(corners,K,T,augmented=True):
    """Clip all 12 box edges to positive optical depth before bounding projection."""
    x=np.c_[corners,np.ones(8)]@np.linalg.inv(T).T
    pts=[p[:3] for p in x if p[2]>=.1]
    for i in range(8):
        for bit in [1,2,4]:
            j=i^bit
            if j<=i:continue
            a,b=x[i],x[j]
            if (a[2]-.1)*(b[2]-.1)<0:
                pts.append((a+(.1-a[2])/(b[2]-a[2])*(b-a))[:3])
    if not pts:return None
    k=A@K if augmented else K
    p=np.asarray(pts)@k.T;uv=p[:,:2]/p[:,2:3]
    bounds=np.r_[uv.min(0),uv.max(0)]
    w,h=(1408,512) if augmented else (640,384)
    bounds=np.clip(bounds,[0,0,0,0],[w,h,w,h])
    return bounds if np.all(bounds[2:]>bounds[:2]) else None

def iou2(a,b):
    if a is None or b is None:return 0.
    a,b=np.asarray(a),np.asarray(b)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=np.prod(wh)
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-12))

def natural(pred,gts,threshold=.1,xy_limit=2.,crop_only=False):
    remaining={i for i,g in enumerate(gts) if not crop_only or g['crop_retained']};den=len(remaining);pairs=[];duplicates=[]
    active=[i for i,s in enumerate(pred['scores']) if s>=threshold]
    for pi in sorted(active,key=lambda i:-pred['scores'][i]):
        b=np.array(pred['boxes'][pi]);kind=coarse(pred['labels'][pi]);possible=[]
        for gi,g in enumerate(gts):
            if crop_only and not g['crop_retained']:continue
            err=np.linalg.norm(b[:2]-np.array(g['box'])[:2])
            if kind==g['type'] and err<=xy_limit and abs(b[2]-g['box'][2])<=2:possible.append((err,gi))
        avail=sorted((e,i) for e,i in possible if i in remaining)
        if avail:gi=avail[0][1];remaining.remove(gi);pairs.append([pi,gi])
        elif possible:duplicates.append(pi)
    return dict(gt=den,predictions=len(active),TP=len(pairs),FP=len(active)-len(pairs),FN=len(remaining),precision=len(pairs)/len(active) if active else None,recall=len(pairs)/den if den else None,duplicates=len(duplicates),pairs=pairs,missed=sorted(remaining))

def conditional_matrix(pred,gts,packet):
    matrix=np.zeros((len(pred['boxes']),len(gts)))
    projected={}
    for ci,c in enumerate(packet['cameras']):
        projected[ci]=[project_bounds(corners_hs(b),np.array(c['K']),np.array(c['T_cam_to_ego'])) for b in pred['boxes']]
    for gi,g in enumerate(gts):
        for pi,label in enumerate(pred['labels']):
            if coarse(label)!=g['type']:continue
            matrix[pi,gi]=max([iou2(projected[v['camera_index']][pi],v['bounds_aug']) for v in g['views'] if v['retained_pixels']>=3] or [0.])
    return matrix

def conditional(pred,gts,matrix,threshold=.1):
    if not matrix.size:return []
    eligible=(matrix>=.1)&(np.array(pred['scores'])[:,None]>=threshold)
    # Maximum cardinality first, then total IoU, deterministic one-to-one.
    cost=np.where(eligible,1000+matrix,0.)
    ii,jj=linear_sum_assignment(-cost)
    return [[int(i),int(j)] for i,j in zip(ii,jj) if eligible[i,j]]

def upright_iou(a,b):
    def poly(v):return Polygon(corners_hs(v)[[0,2,6,4],:2])
    pa,pb=poly(a),poly(b);area=pa.intersection(pb).area
    height=max(0,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    inter=area*height;return float(inter/(pa.area*a[5]+pb.area*b[5]-inter+1e-12))

def errors(pred,gts,pairs,packet):
    rows=[]
    for pi,gi in pairs:
        b=np.array(pred['boxes'][pi]);g=gts[gi];gt=np.array(g['box']);d=b[:3]-gt[:3];v=max(g['views'],key=lambda v:v['retained_pixels']);T=np.array(packet['cameras'][v['camera_index']]['T_cam_to_ego']);opt=np.linalg.inv(T)[:3,:3]@d
        yaw=float(abs((b[6]-gt[6]+np.pi)%(2*np.pi)-np.pi))
        rows.append(dict(prediction_index=pi,gt_index=gi,actor_id=g['actor_id'],score=pred['scores'][pi],type=g['type'],xy=float(np.linalg.norm(d[:2])),optical_depth_abs=float(abs(opt[2])),optical_lateral_abs=float(abs(opt[0])),radial_abs=float(abs(np.linalg.norm(b[:2])-np.linalg.norm(gt[:2]))),dimension_abs=np.abs(b[3:6]-gt[3:6]).tolist(),yaw_abs_deg=float(np.degrees(yaw)),yaw_axis_abs_deg=float(np.degrees(min(yaw,np.pi-yaw))),upright_iou3d=upright_iou(b,gt),gt_range=float(np.linalg.norm(gt[:2])),gt_upright_approx_corner_error=g['upright_corner_error_m']))
    return rows

def aggregate_natural(results):
    total={k:sum(r[k] for r in results) for k in ['gt','predictions','TP','FP','FN','duplicates']}
    total['precision']=total['TP']/total['predictions'] if total['predictions'] else None
    total['recall']=total['TP']/total['gt'] if total['gt'] else None
    return total
