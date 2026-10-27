"""Fixed training-set diagnostics; decoded predictions never use GT filtering."""
import numpy as np


def image_panels(array):
    """PETR may squeeze batch=1 in place; visualize either shape unchanged."""
    if array.ndim == 5:
        assert array.shape[0] == 1
        array = array[0]
    assert array.ndim == 4 and array.shape[:2] == (6, 3)
    return array.transpose(0, 2, 3, 1) + np.array([103.530,116.280,123.675])


def prediction_mask(boxes, scores, threshold):
    return (np.abs(boxes[:, 0]) < 51.2) & (np.abs(boxes[:, 1]) < 51.2) & (scores >= threshold)


def match(boxes, scores, labels, gt, gt_labels, threshold=0.2, distance=2.0):
    """Score-ordered same-class one-to-one; unmatched predictions remain FP."""
    indices = np.where(prediction_mask(boxes, scores, threshold))[0]
    indices = sorted(indices, key=lambda i: (-float(scores[i]), int(i)))
    taken = set(); pairs = []; fp = []; duplicates = []
    for i in indices:
        same = np.where(gt_labels == labels[i])[0]
        choices = [j for j in same if j not in taken]
        if choices:
            j = min(choices, key=lambda j: (float(np.linalg.norm(boxes[i, :2]-gt[j, :2])), int(j)))
            if np.linalg.norm(boxes[i, :2]-gt[j, :2]) < distance:
                taken.add(j); pairs.append((i, j)); continue
        fp.append(i)
        if any(np.linalg.norm(boxes[i, :2]-gt[j, :2]) < distance for j in same):
            duplicates.append(i)
    return dict(pairs=pairs, fp=fp, fn=[i for i in range(len(gt)) if i not in taken],
                duplicates=duplicates, selected=indices)


def distribution(values):
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None, p95=None, max=None)
    a=np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a,90)), p95=float(np.percentile(a,95)), max=float(a.max()))


def errors(boxes, gt, pairs, gt_labels, classes):
    result={k:[] for k in ('xy_m','z_center_m','dimension_abs_mean_m','aligned_size_iou_error','yaw_rad')}
    for i,j in pairs:
        p,g=boxes[i],gt[j]
        result['xy_m'].append(float(np.linalg.norm(p[:2]-g[:2])))
        result['z_center_m'].append(float(abs(p[2]+p[5]/2-g[2]-g[5]/2)))
        result['dimension_abs_mean_m'].append(float(np.abs(p[3:6]-g[3:6]).mean()))
        intersection=float(np.minimum(p[3:6],g[3:6]).prod())
        result['aligned_size_iou_error'].append(1-intersection/(p[3:6].prod()+g[3:6].prod()-intersection))
        name=classes[int(gt_labels[j])]
        if name!='traffic_cone':
            period=np.pi if name=='barrier' else 2*np.pi
            result['yaw_rad'].append(float(abs((p[6]-g[6]+period/2)%period-period/2)))
    return result


def summarize(records, classes, threshold=0.2, distance=2.0):
    counts=[dict(GT=0,TP=0,FP=0,FN=0,duplicates=0) for _ in classes]
    errs=[{k:[] for k in ('xy_m','z_center_m','dimension_abs_mean_m','aligned_size_iou_error','yaw_rad')} for _ in classes]
    decoded=outside=selected=0
    for r in records:
        b,s,l,g,gl=[r[k] for k in ('boxes','scores','labels','gt','gt_labels')]
        m=match(b,s,l,g,gl,threshold,distance)
        decoded+=len(b); outside+=int((~prediction_mask(b,s,0.)).sum()); selected+=len(m['selected'])
        for ci in range(len(classes)):
            q=counts[ci]; pairs=[(i,j) for i,j in m['pairs'] if gl[j]==ci]
            q['GT']+=int((gl==ci).sum()); q['TP']+=len(pairs)
            q['FP']+=sum(l[i]==ci for i in m['fp']);q['FN']+=sum(gl[j]==ci for j in m['fn'])
            q['duplicates']+=sum(l[i]==ci for i in m['duplicates'])
            for k,v in errors(b,g,pairs,gl,classes).items():errs[ci][k].extend(v)
    def rates(q):
        q={k:int(v) for k,v in q.items()}
        return dict(q,precision=q['TP']/(q['TP']+q['FP']) if q['TP']+q['FP'] else None,
                    recall=q['TP']/q['GT'] if q['GT'] else None)
    aggregate={k:sum(q[k] for q in counts) for k in counts[0]}
    return dict(score_threshold=threshold,distance_m=distance,
        **rates(aggregate),decoded_predictions=decoded,outside_task_range_predictions=outside,
        selected_predictions=selected,matched_errors={k:distribution([x for e in errs for x in e[k]]) for k in errs[0]},
        per_class={name:dict(**rates(q),matched_errors={k:distribution(v) for k,v in e.items()})
                   for name,q,e in zip(classes,counts,errs)})


def average_precision(records, classes, distance=2.):
    result={}
    for ci,name in enumerate(classes):
        entries=[];ngt=0
        for fi,r in enumerate(records):
            m=match(r['boxes'],r['scores'],r['labels'],r['gt'],r['gt_labels'],0.,distance)
            positive={i for i,_ in m['pairs']};ngt+=int((r['gt_labels']==ci).sum())
            entries.extend((float(r['scores'][i]),fi,int(i),i in positive)
                           for i in m['selected'] if r['labels'][i]==ci)
        entries.sort(key=lambda x:(-x[0],x[1],x[2]))
        if not ngt:
            result[name]=dict(GT=0,AP=None,predictions=len(entries),reason='NoGT_NotApplicable');continue
        tp=np.cumsum([e[3] for e in entries]);fp=np.cumsum([not e[3] for e in entries])
        recall=tp/ngt;precision=tp/np.maximum(tp+fp,1)
        ap=float(np.mean([precision[recall>=v].max() if np.any(recall>=v) else 0.
                          for v in np.linspace(0,1,101)]))
        result[name]=dict(GT=ngt,AP=ap,predictions=len(entries),max_recall=float(recall[-1]) if len(recall) else 0.)
    valid=[r['AP'] for r in result.values() if r['AP'] is not None]
    return dict(per_class=result,mean_supported_classes=float(np.mean(valid)) if valid else None,
                definition='101PointInterpolatedTrainingDiagnostic_NotOfficialNuScenesMAP')
