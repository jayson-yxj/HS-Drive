"""Low-FPR workpoints and visible connected-component false detections.

No unobserved pixel is a negative. Components are image detections, not independent
physical Actors; Actor-level recall uses associated visible GT, >=3 pixels.
"""
import numpy as np,cv2
from diagnostics.m5f_metrics import histogram,metrics
from tools.m5g_evaluation import summarize

def enhanced(rows,labels,probs,threshold=None,classes=(1,2,3),locked_low=None):
    result=summarize(rows,labels,probs,threshold,classes);h=np.array(result['histogram']);m=result['metrics'];valid_n=int(h.sum());fp=m['fp'];result['confirmed_nontarget_fpr']=m['fpr'];result['false_foreground_per_million_valid_pixels']=fp/max(valid_n,1)*1e6;curves={};false_components=[]
    for row,d,p in zip(rows,labels,probs):
        target=d['target']&np.isin(d['classes'],classes);valid=d['valid']&(~d['target']|target);cut=int(result['threshold']*256)/256
        for c,name in enumerate(row['camera_names']):
            for key,mask in [(f'scene:{row["group"]}',valid[c]),(f'camera:{name}',valid[c])]+[(f'distance:{lo}-{hi}',valid[c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi))for lo,hi in [(0,10),(10,20),(20,35),(35,80)]]:
                hh=histogram(p[c],target[c],mask);curves[key]=curves.get(key,np.zeros((2,256),np.int64))+hh
            binary=((p[c]>=cut)&valid[c]).astype(np.uint8);n,cc,stats,_=cv2.connectedComponentsWithStats(binary,8);fpcount=tpcount=0
            for k in range(1,n):
                if stats[k,cv2.CC_STAT_AREA]<3:continue
                if np.any((cc==k)&target[c]):tpcount+=1
                else:fpcount+=1
            false_components.append(dict(group=row['group'],sample_id=row['sample_id'],camera=name,false_components=fpcount,components_intersect_target=tpcount))
    grid=[k/256 for k in range(1,257)];result['pr_curves']={key:[dict(threshold=t,**metrics(hh,t))for t in grid]for key,hh in curves.items()};result['false_detection_components']=false_components
    observable=[a for a in result['actors']if a['pixels']>=3];result['actor_miss_rate']=sum(not a['detected']for a in observable)/len(observable)if observable else None;result['observable_actor_views']=len(observable)
    if locked_low is None:
        curve=[dict(threshold=t,**metrics(h,t))for t in grid];low={}
        for fpr in [.01,.001,.0001]:
            eligible=[x for x in curve if (x['fpr']or 0)<=fpr];low[str(fpr)]=max(eligible,key=lambda x:((x['recall']or 0),x['threshold']))if eligible else None
        result['low_fpr_validation']=low
    else:result['low_fpr_locked_evaluation']={k:dict(threshold=v['threshold'],**metrics(h,v['threshold']))if v else None for k,v in locked_low.items()}
    result['metric_limits']='False components>=3 known pixels disconnectedbyUnknown; not physicalActor precision. Curves descriptive; lowFPRthreshold selectedonlyVal. Differentactorviews/scenes correlated. Originaljointgates unchanged.'
    return result
