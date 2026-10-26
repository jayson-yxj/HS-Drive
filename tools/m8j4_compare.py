"""Post-inference evaluator only: compare common class/geometry, no GT feedback."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from tools.m8j4_sidecar import OUT,save
from tools.m8j3_validate_clip import RAW,read_annotation


def corners(box):
    b=np.asarray(box)
    offsets=np.array([[x,y,z] for x in [.5,-.5] for y in [.5,-.5] for z in [.5,-.5]])*b[3:6]
    c,s=np.cos(b[6]),np.sin(b[6]);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return offsets@R.T+b[:3]


def main():
    out=OUT/'comparison';out.mkdir(exist_ok=False)
    summaries=[];details=[]
    for ordinal in ('00000','00107','00213'):
        frame=json.loads((OUT/'reviewed_sidecar'/(ordinal+'.json')).read_text())
        pred=json.loads((OUT/'forward'/(ordinal+'.json')).read_text())
        gt=[]
        for obj in frame['objects']:
            if not obj['positive_supervision_candidate']:continue
            center=np.asarray(obj['center_hs_m'])
            inrange=bool(np.all(center>[-51.2,-51.2,-5]) and np.all(center<[51.2,51.2,3]))
            retained=False
            for v in obj['views'].values():
                if v['status']=='Valid':
                    b=v['projection']['bbox_xyxy']
                    # Actual official image crop after resize .88 and y offset 280.
                    retained |= b[3]*.88>280 and b[1]*.88<792 and b[2]>0 and b[0]<1600
            if inrange and retained:gt.append(obj)
        boxes=np.asarray(pred['boxes_hs']);scores=np.asarray(pred['scores']);labels=np.asarray(pred['labels'])
        keep=np.flatnonzero(scores>=.1)
        # Descriptive nearest same-class predictions, not detections or an AP estimate.
        nearest=[]
        for g in gt:
            cid=pred['label_names'].index(g['model_class']);ids=keep[labels[keep]==cid]
            if not len(ids):
                nearest.append(dict(actor_id=g['actor_id'],status='NoSameClassPredictionAtFixedScore'));continue
            distances=np.linalg.norm(boxes[ids,:2]-np.asarray(g['center_hs_m'])[:2],axis=1)
            j=ids[np.argmin(distances)]
            nearest.append(dict(actor_id=g['actor_id'],status='NearestOnly_NotAssociation',
                prediction_index=int(j),xy_error_m=float(distances.min()),score=float(scores[j]),
                dimensions_abs_error_m=abs(boxes[j,3:6]-g['dimensions_lwh_m']).tolist(),
                yaw_abs_error_rad=float(abs((boxes[j,6]-g['yaw_ccw_rad']+np.pi)%(2*np.pi)-np.pi))))
        summary=dict(ordinal=ordinal,gt_common_positive_count=len(gt),candidate_count=len(boxes),
            at_fixed_score_0_1=len(keep),candidate_class_counts=dict(Counter(pred['label_names'][labels[i]] for i in keep)),
            score_min=float(scores.min()),score_max=float(scores.max()),
            comparable_fields=['HS ego ActorBoxCenter','full lwh','CCW yaw','car/pedestrian'],
            common_region='PETR pc_range AND a supported view intersects actual resized bottom crop',
            unmatched_predictions='UnknownLabelCompleteness_NotAutomaticallyFalsePositive',
            excluded=['Unknown GT','underground','unsupported traffic lights/signs','outside range/crop','velocity'],
            nearest_diagnostic_count=len(nearest),accuracy_certified=False)
        summaries.append(summary);details.append(dict(**summary,nearest=nearest))
        # Private 2D BEV comparison, green reviewed GT, red score>=.1 predictions.
        im=np.full((820,820,3),245,np.uint8)
        def pixel(p):return (int(410-p[1]*7),int(410-p[0]*7))
        for i in keep:
            if labels[i] not in (0,8):continue
            p=boxes[i,:2]
            if abs(p).max()<55:cv2.circle(im,pixel(p),3,(20,20,220),-1)
        for g in gt:
            p=np.asarray(g['center_hs_m']);cv2.circle(im,pixel(p),7,(0,140,0),2)
            cv2.putText(im,g['actor_id'],pixel(p),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,100,0),1)
        cv2.drawMarker(im,(410,410),(0,0,0),cv2.MARKER_TRIANGLE_UP,14,2)
        cv2.putText(im,ordinal+' green=review GT red=prediction>=0.1',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        cv2.putText(im,'diagnostic only; unlabelled space is Unknown',(10,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        cv2.imwrite(str(out/(ordinal+'_bev.jpg')),im)
    save(out/'private_comparison.json',details)
    save(Path('reports/m8j4_prediction_comparability.json'),dict(status='CoordinateClassComparisonPossible_QualityNotCertified',
        records=summaries,score_threshold_from_existing_M8B=.1,no_selection_or_threshold_tuning=True,
        precision_recall_AP='NotReported_IncompleteLabels_ThreeCorrelatedFrames'))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
