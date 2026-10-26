"""Supervision-only image coverage. Render tokens never identify Actors.

Image masks can veto a negative; they cannot certify a 3D negative or create a
positive Box. This module is independent of torch and the RGB inference loader.
"""
import cv2
import numpy as np

from datasets.m8j4_gt import bbox_iou

FOREGROUND_IDS = (12, 13, 14, 15, 16, 17, 18, 19)


def projection_mask(projection, shape):
    out = np.zeros(shape, np.uint8)
    if projection.get('status') == 'Valid':
        cv2.fillConvexPoly(out, np.rint(projection['polygon']).astype(np.int32), 1)
    return out.astype(bool)


def coverage_view(semantic, instance, objects, camera, dilation=4):
    if semantic.dtype != np.uint8 or semantic.ndim != 2:
        raise ValueError('Expected raw semantic uint8 labels, not palette RGB')
    if instance.dtype != np.uint8 or instance.shape != (*semantic.shape, 4):
        raise ValueError('Expected raw BGRA instance encoding')
    if dilation < 0:
        raise ValueError('Negative dilation')
    tags = instance[:, :, 2]
    tokens = instance[:, :, 1].astype(np.int32) + 256 * instance[:, :, 0].astype(np.int32)
    fg = np.isin(semantic, FOREGROUND_IDS) | np.isin(tags, FOREGROUND_IDS)
    support = np.zeros_like(fg)
    for obj in objects:
        cls = obj['source_class']
        if cls not in ('vehicle', 'walker') or obj['converted']['geometry_status'] != 'Valid':
            continue
        ids = (12,) if cls == 'walker' else (14, 15, 16, 17, 18, 19)
        # Union support is geometry, not association, and does not erase labels.
        projection = obj['views'].get(camera, {}).get('raw_projection', {})
        support |= projection_mask(projection, semantic.shape) & np.isin(tags, ids)
    mismatch = (semantic != tags) & fg
    uncovered = fg & ~support
    missing = np.zeros_like(fg)
    records = []
    for label in FOREGROUND_IDS:
        yy, xx = np.nonzero(tags == label)
        if not len(xx):
            continue
        ids, inverse, sizes = np.unique(tokens[yy, xx], return_inverse=True, return_counts=True)
        hits = np.bincount(inverse, weights=support[yy, xx], minlength=len(ids))
        agreement = np.bincount(inverse, weights=(semantic[yy, xx] == label), minlength=len(ids))
        for k, token in enumerate(ids):
            selected = inverse == k
            y, x = yy[selected], xx[selected]
            fraction = float(hits[k] / sizes[k])
            candidate = fraction < .5
            if candidate:
                missing[y, x] = True
            records.append(dict(semantic=int(label), render_token=int(token), pixels=int(sizes[k]),
                raw_support_fraction=fraction, semantic_agreement=float(agreement[k]/sizes[k]),
                bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                status='MissingSourceSupportCandidate' if candidate else 'RawGeometrySupport_NotIdentity',
                actor_identity='Unknown', reliable_3d_positive=False))
    # Train/rider and non-car vehicle semantics are not exhaustively annotated;
    # disagreement and semantic-only pixels remain unknown even without token.
    unsupported = np.isin(tags, (13, 15, 16, 17, 18, 19)) | np.isin(semantic, (13, 15, 16, 17, 18, 19))
    seed = uncovered | missing | mismatch | unsupported
    ignore = cv2.dilate(seed.astype(np.uint8), np.ones((2*dilation+1, 2*dilation+1), np.uint8)).astype(bool)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ignore.astype(np.uint8), 8)
    rectangles = [dict(bbox_xyxy=[int(x), int(y), int(x+w), int(y+h)], pixels=int(area))
                  for x, y, w, h, area in stats[1:]]
    return dict(ignore=ignore, missing=missing, support=support, foreground=fg, records=records,
        rectangles=rectangles, stats=dict(pixels=int(fg.size), foreground_pixels=int(fg.sum()),
        agreed_foreground_pixels=int((fg & (semantic == tags)).sum()),
        full_image_agreement_pixels=int((semantic == tags).sum()),
        raw_supported_pixels=int(support.sum()), uncovered_pixels=int(uncovered.sum()),
        missing_region_pixels=int(missing.sum()), disagreement_foreground_pixels=int(mismatch.sum()),
        ignore_pixels=int(ignore.sum()), ignore_components=count-1,
        safe_negative_pixels_certified=0))


def positive_eligibility(obj, regions, camera_order):
    """Source Box validity + class + conservative visual support; identity separate.

    Eligibility is for a loss diagnostic, not Golden GT or training approval.
    Render ambiguity does not turn valid Box algebra into invalid geometry.
    """
    reasons=[];old=obj.get('legacy') or {};new=obj['converted'];raw=obj['raw']
    cls = 'pedestrian' if obj['source_class']=='walker' else ('car' if obj['source_class']=='vehicle' and raw.get('base_type')=='car' else None)
    if new['geometry_status']!='Valid' or old.get('geometry_status')!='Valid': reasons.append('GeometryUnreliable')
    if cls is None: reasons.append('ClassUnsupported')
    c=np.asarray(new.get('center_hs_m',[np.nan]*3))
    if not (np.all(c>[-51.2,-51.2,-5]) and np.all(c<[51.2,51.2,3])): reasons.append('OutsideModelRange')
    if not old.get('yaw_only_comparable',False): reasons.append('YawOnlyRepresentationUnsupported')
    if old.get('reason')=='HiddenBelowGround_NoTrainingTarget': reasons.append('UndergroundSourceRecord')
    support=[]
    expected=12 if cls=='pedestrian' else 14
    for cam in camera_order:
        projection=obj['views'].get(cam,{}).get('raw_projection',{})
        if projection.get('status')!='Valid' or projection.get('near_plane_clipped'):continue
        b=projection['bbox_xyxy']
        if b[3]*.88<=280 or b[1]*.88>=792:continue
        for r in regions:
            if r['camera']!=cam or r['semantic']!=expected or r['pixels']<12 or r['semantic_agreement']<.9:continue
            if r['bbox_xyxy'][3]*.88<=280 or r['bbox_xyxy'][1]*.88>=792:continue
            hit=next((h for h in r['raw_box_overlap_candidates'] if h['actor_id']==obj['actor_id']),None)
            if hit and hit['pixels']>=12 and hit['fraction']>=.8 and bbox_iou(b,r['bbox_xyxy'])>=.25:
                support.append(cam);break
    if not support: reasons.append('VisibleSemanticSupportUnconfirmedInCrop')
    return dict(eligible=not reasons,model_class=cls,reasons=reasons,supported_cameras=support,
        geometry_validity=new['geometry_status'],identity_validity='NotRequiredForSourceBoxGeometry',
        human_review='Pending_Suspended',golden_gt=False,authorization='LossDiagnosticOnly')
