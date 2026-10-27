"""Pure NumPy diagnostics of frozen decoded/raw caches. Never runs a model."""
from collections import defaultdict
import numpy as np
from training.m8j7_metrics import match, prediction_mask


def sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -x))


def match_record(record, threshold=.2):
    return match(**{k:record[k] for k in ('boxes','scores','labels','gt','gt_labels')},
                 threshold=threshold, distance=2.)


def exact_curve(records):
    """All realizable threshold points on the ORIGINAL decoded output.

    Greedy high-score assignment is a prefix operation: adding lower scores
    cannot change prior assignments. Equal scores must be included together.
    No raw query is inserted into the scored prediction set.
    """
    entries = []
    total_gt = sum(len(r['gt']) for r in records)
    for fi, r in enumerate(records):
        matched = match_record(r, 0.)
        positive = {i for i, _ in matched['pairs']}
        duplicates = set(matched['duplicates'])
        for i in matched['selected']:
            entries.append((float(r['scores'][i]), fi, int(i), i in positive, i in duplicates))
    entries.sort(key=lambda e:(-e[0], e[1], e[2]))
    points = [];tp=fp=dup=0;offset=0
    while offset < len(entries):
        score = entries[offset][0]
        end = offset
        while end < len(entries) and entries[end][0] == score:
            tp += int(entries[end][3]);fp += int(not entries[end][3]);dup += int(entries[end][4]);end += 1
        points.append(dict(score_threshold=score,TP=tp,FP=fp,FN=total_gt-tp,GT=total_gt,
                           duplicates=dup,precision=tp/(tp+fp),recall=tp/total_gt if total_gt else None))
        offset = end
    return points


def operating_points(curve, reference):
    reaching = next((i for i,p in enumerate(curve) if p['TP'] >= reference['TP']), None)
    closest = min(curve, key=lambda p:(abs(p['precision']-reference['precision']), -p['score_threshold'])) if curve else None
    return dict(
        target_reference=reference,
        recall_target_reachable=reaching is not None,
        recall_first_reaching=None if reaching is None else curve[reaching],
        recall_immediately_below=None if reaching is None or reaching == 0 else curve[reaching-1],
        nearest_observed_precision=closest,
        maximum_cache_recall=curve[-1] if curve else None,
        interpretation='PostHocMechanismOnly_NoInterpolation_NoThresholdReplacement_NoTopKExpansion')


def summarize_transitions(rows):
    names=('common','A_only','B_only','neither')
    def counts(items):
        return dict(records=len(items),unique_instances=len({r['instance_token'] for r in items}),
                    states={s:dict(records=sum(r['state']==s for r in items),
                        unique_instances=len({r['instance_token'] for r in items if r['state']==s})) for s in names})
    return dict(total=counts(rows),
        scenes={s:counts([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},
        classes={c:counts([r for r in rows if r['class_name']==c]) for c in sorted({r['class_name'] for r in rows})},
        scene_class={s:{c:counts([r for r in rows if r['scene']==s and r['class_name']==c])
                       for c in sorted({r['class_name'] for r in rows if r['scene']==s})}
                     for s in sorted({r['scene'] for r in rows})},
        unique_instance_caveat='State instance sets may overlap across time; do not add them as disjoint physical targets.')


def select_cases(rows, config):
    used=set();chosen=[];visuals=set(config['existing_visual_frames'])
    for bucket in config['bucket_order']:
        eligible=[r for r in rows if r['state']==bucket or
                  (bucket=='common_xy_worse' and r['state']=='common' and r['xy_delta_m']>=config['common_xy_worse_min_delta_m'])]
        representatives={}
        for r in sorted(eligible,key=lambda r:(r['frame'] not in visuals,r['timestamp_us'],r['sample_token'])):
            representatives.setdefault(r['instance_token'],r)
        pool=list(representatives.values());used_classes=set();cursor=0
        for _ in range(config['budgets'][bucket]):
            available=[r for r in pool if r['instance_token'] not in used]
            if not available:break
            scene=config['scenes'][cursor%len(config['scenes'])];cursor+=1
            scoped=[r for r in available if r['scene']==scene] or available
            candidate=min(scoped,key=lambda r:(r['class_name'] in used_classes,r['frame'] not in visuals,
                r['class_name'],r['instance_token'],r['timestamp_us'],r['sample_token']))
            used.add(candidate['instance_token']);used_classes.add(candidate['class_name'])
            chosen.append(dict(candidate,bucket=bucket,case_id=f'C{len(chosen)+1:02}'))
    assert len(chosen)<=config['maximum_unique_instances']
    assert len({r['instance_token'] for r in chosen})==len(chosen)
    return chosen


def query_observation(record, gt_index, query):
    g=record['gt'][gt_index];ci=int(record['gt_labels'][gt_index]);q=int(query)
    probabilities=sigmoid(record['raw_cls'][q])
    return dict(query=q,correct_class_score=float(probabilities[ci]),
        argmax_class=int(np.argmax(probabilities)),argmax_score=float(probabilities.max()),
        xy_distance_m=float(np.linalg.norm(record['raw_bbox'][q,:2]-g[:2])),
        center_z_error_m=float(record['raw_bbox'][q,4]-(g[2]+g[5]/2)),
        correct_class_decoded=(q*len(probabilities)+ci) in set(record['flat_indices'].tolist()))


def evidence(record, gt_index, matching):
    """GT-local candidate evidence, never a new inference-selection method."""
    j=int(gt_index);g=record['gt'][j];ci=int(record['gt_labels'][j]);raw=record['raw_bbox']
    probs=sigmoid(record['raw_cls']);dist=np.linalg.norm(raw[:,:2]-g[:2],axis=1)
    centers=raw[:,[0,1,4]]
    decoder_ok=((centers>=np.array([-61.2,-61.2,-10]))&(centers<=np.array([61.2,61.2,10]))).all(1)
    task_ok=(np.abs(raw[:,0])<51.2)&(np.abs(raw[:,1])<51.2)
    near=dist<2.;near_valid=near&decoder_ok&task_ok
    raw_candidates=np.flatnonzero(near_valid)
    best=int(max(raw_candidates,key=lambda q:(probs[q,ci],-int(q)))) if len(raw_candidates) else None
    nearby_decoded=[int(i) for i in range(len(record['boxes'])) if int(record['labels'][i])==ci
        and np.linalg.norm(record['boxes'][i,:2]-g[:2])<2.
        and prediction_mask(record['boxes'][i:i+1],record['scores'][i:i+1],0.)[0]]
    above=[i for i in nearby_decoded if record['scores'][i]>=.2]
    by_gt={gi:pi for pi,gi in matching['pairs']};by_pred={int(pi):int(gi) for pi,gi in matching['pairs']}
    pi=by_gt.get(j)
    raw_flat=np.arange(len(raw))*probs.shape[1]+ci
    kept=np.isin(raw_flat,record['flat_indices'])
    strong=near_valid&(probs[:,ci]>=.2)
    flat_scores=probs.reshape(-1)
    cutoff=float(np.partition(flat_scores,-300)[-300])
    # Distinguish rank rejection from decoder range. No tie-dependent topK identity inference.
    excluded=[int(q) for q in np.flatnonzero(strong&~kept)]
    flags=[]
    if pi is None:
        if above:flags.append('QualifiedDecodedCandidateConsumedByOneToOneCompetition')
        if nearby_decoded and not above:flags.append('NearbyCorrectClassDecodedScoresBelow0.2')
        if excluded:flags.append('NearbyCorrectClassScoreAtLeast0.2ExcludedFromTop300')
        if not near.any():flags.append('NoRawQueryCenterWithin2m')
        elif not near_valid.any():flags.append('NearbyRawCentersExcludedByExistingRange')
        if near_valid.any() and probs[near_valid,ci].max()<.2:flags.append('AllNearbyValidCorrectClassRawScoresBelow0.2')
        if best is not None and probs[best].argmax()!=ci:flags.append('BestNearbyCorrectClassSlotHasDifferentArgmaxClass_NotExclusiveClassifier')
    matched=None
    if pi is not None:
        p=record['boxes'][pi]
        matched=dict(prediction_index=int(pi),query=int(record['query_indices'][pi]),score=float(record['scores'][pi]),
            xy_error_m=float(np.linalg.norm(p[:2]-g[:2])),center_z_error_m=float(p[2]+p[5]/2-g[2]-g[5]/2),
            height_error_m=float(p[5]-g[5]),dimension_abs_mean_m=float(np.abs(p[3:6]-g[3:6]).mean()))
    return dict(matched=matched,flags=flags,nearest_raw=query_observation(record,j,int(dist.argmin())),
        best_correct_class_nearby=None if best is None else query_observation(record,j,best),
        raw_query_count_near=int(near.sum()),raw_query_count_near_in_range=int(near_valid.sum()),
        strong_nearby_top300_excluded_queries=excluded,
        strong_nearby_top300_exclusion_count=len(excluded),top300_min_class_score=cutoff,
        near_decoded=[dict(prediction_index=i,query=int(record['query_indices'][i]),score=float(record['scores'][i]),
            xy_error_m=float(np.linalg.norm(record['boxes'][i,:2]-g[:2])),assigned_GT=by_pred.get(i)) for i in nearby_decoded],
        cache_only=True,inference_selection_unchanged=True)
