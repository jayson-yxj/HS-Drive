"""Paired, evaluator-only metrics; strict image identity precedes metric errors."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from diagnostics.m8b_metrics import project_bounds, iou2, upright_iou
from diagnostics.m8c_geometry import association_grade, pixel_support
from datasets.m8b_petr_adapter import ORDER, corners_hs


def load_masks(frame, index, cache):
    row = index['row']
    raw = json.loads(Path(row['raw_frame']).read_text())
    base = Path(row['raw_frame']).parent.parent
    entries = {(a['actor_id'], a['camera_index']): a for a in cache['actors']
               if a['sample_id'] == frame['sample_id'] and a['seed'] == 101}
    with np.load(row['pixel_label'], allow_pickle=False) as data:
        depth = data['depth']
    masks, visible, tight = {}, {}, {}
    for ci in range(6):
        if not any((g['actor_id'],ci) in entries for g in frame['gts']):
            continue
        with np.load(base/raw['cameras'][ORDER[ci]]['instance'], allow_pickle=False) as data:
            instance = data['instance']
        actors = row['actors'][ci]['actors']
        claims = Counter(q['render_id'] for a in actors for q in a['identity_candidates'] if q['accepted'])
        for gi,g in enumerate(frame['gts']):
            key = (g['actor_id'],ci)
            if key not in entries:
                continue
            actor = next(a for a in actors if a['actor_id']==g['actor_id'])
            ids = [q['render_id'] for q in actor['identity_candidates']
                   if q['accepted'] and claims[q['render_id']]==1]
            mask = np.isin(instance,ids)&np.isfinite(depth[ci])&(depth[ci]>=1.5)&(depth[ci]<80)
            assert mask.sum()==entries[key]['native_visible_pixels']
            mask[:151]=False
            masks[(gi,ci)]=mask
            tight[(gi,ci)]=pixel_support(mask,None)['bbox']
            visible[(gi,ci)]=entries[key]
    return masks,tight,visible


def metric_error(box, gt, packet, views):
    b=np.asarray(box);g=np.asarray(gt['box']);delta=b[:3]-g[:3]
    ci=max(views,key=lambda v:v['retained_pixels'])['camera_index']
    optical=np.linalg.inv(packet['cameras'][ci]['T_cam_to_ego'])[:3,:3]@delta
    yaw=abs((b[6]-g[6]+np.pi)%(2*np.pi)-np.pi)
    return dict(XY_m=float(np.linalg.norm(delta[:2])),XYZ_m=float(np.linalg.norm(delta)),
        optical_depth_signed_m=float(optical[2]),optical_lateral_signed_m=float(optical[0]),
        radial_signed_m=float(np.linalg.norm(b[:2])-np.linalg.norm(g[:2])),
        center_Z_signed_m=float(delta[2]),dimensions_abs_lwh_m=np.abs(b[3:6]-g[3:6]).tolist(),
        yaw_absolute_deg=float(np.degrees(yaw)) if gt['type']!='cone' else None,
        yaw_axis_deg=float(np.degrees(min(yaw,np.pi-yaw))) if gt['type']!='cone' else None,
        upright_3d_iou=upright_iou(b,g),best_view=ORDER[ci])


def grade_frame(pred, frame, packet, pairs, support):
    masks,tight,visible=support
    paired={gi:pi for pi,gi in pairs}
    rows=[]
    for gi,g in enumerate(frame['gts']):
        pi=paired.get(gi);views=[]
        if pi is not None:
            for ci in range(6):
                if (gi,ci) not in masks:
                    continue
                c=packet['cameras'][ci]
                bounds=project_bounds(corners_hs(pred['boxes'][pi]),np.asarray(c['K']),
                                      np.asarray(c['T_cam_to_ego']),augmented=False)
                if bounds is not None:
                    bounds=bounds.copy();bounds[1]=max(bounds[1],332/2.2)
                    if bounds[3]<=bounds[1]:bounds=None
                p=pixel_support(masks[(gi,ci)],bounds)
                competitors=[iou2(bounds,b) for (gj,cj),b in tight.items() if cj==ci and gj!=gi]
                views.append(dict(camera=ORDER[ci],camera_index=ci,retained_pixels=p['pixels'],
                    mask_recall=p['mask_recall'],instance_bbox_iou=p['bbox_iou'],
                    GT_association_purity=visible[(gi,ci)]['association_purity'],
                    competing_bbox_iou=max(competitors or [0.]),
                    projected_bbox=None if bounds is None else bounds.tolist(),instance_bbox=p['bbox']))
        grade=association_grade(views,pi is not None)
        rows.append(dict(token=frame['token'],gt_index=gi,actor_id=g['actor_id'],
            frame_id=packet['frame_id'],timestamp_s=packet['timestamp_s'],
            split=frame['split'],group=frame['group'],type=g['type'],
            range_m=float(np.linalg.norm(g['box'][:2])),grade=grade,prediction_index=pi,
            score=None if pi is None else pred['scores'][pi],views=views,
            metric_error=metric_error(pred['boxes'][pi],g,packet,views) if grade=='IdentitySupported' else None))
    return rows


def transition(a,b):
    return ('A_success' if a else 'A_failure')+'__'+('B_success' if b else 'B_failure')


def paired_counts(rows):
    counts=Counter(transition(r['A'],r['B']) for r in rows)
    n=len(rows)
    return dict(GT=n,A_TP=sum(r['A'] for r in rows),B_TP=sum(r['B'] for r in rows),
                A_recall=sum(r['A'] for r in rows)/n if n else None,
                B_recall=sum(r['B'] for r in rows)/n if n else None,
                delta_TP=sum(int(r['B'])-int(r['A']) for r in rows),
                transitions={transition(a,b):counts[transition(a,b)] for a in [True,False] for b in [True,False]})


def localization(rows, stats):
    errors=[r['metric_error'] for r in rows if r['metric_error'] is not None]
    keys=['XY_m','XYZ_m','optical_depth_signed_m','optical_lateral_signed_m',
          'radial_signed_m','center_Z_signed_m','yaw_absolute_deg','yaw_axis_deg','upright_3d_iou']
    return dict(n=len(errors),groups=len({r['group'] for r in rows if r['metric_error'] is not None}),
        statistics={k:stats([e[k] for e in errors if e[k] is not None]) for k in keys},
        depth_abs=stats([abs(e['optical_depth_signed_m']) for e in errors]),
        lateral_abs=stats([abs(e['optical_lateral_signed_m']) for e in errors]),
        dimension_abs_lwh=[stats([e['dimensions_abs_lwh_m'][i] for e in errors]) for i in range(3)],
        over5_rate=sum(e['XY_m']>5 for e in errors)/len(errors) if errors else None)
