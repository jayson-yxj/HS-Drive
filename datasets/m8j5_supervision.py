"""Independent GT evidence axes and human review gate. Never an inference input."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def classify_object(obj):
    geometry = obj['geometry_status']
    cls = 'Valid' if obj.get('model_class') in ('car', 'pedestrian') else 'Unknown'
    views = obj.get('views', {})
    supported = [v for v in views.values() if v['status'] == 'Valid']
    conflicts = any('withdrawn_match' in v or 'SharedRender' in v.get('reason', '') for v in views.values())
    identity = 'Unknown' if conflicts or not supported else 'EvidenceSupported_NotHumanVerified'
    projected = [v for v in views.values() if v.get('projection', {}).get('status') == 'Valid']
    if supported:
        visibility = 'VisibleEvidence_PartialOcclusionPossible' if any('PartialOcclusion' in v.get('occlusion_state', '') for v in supported) else 'VisibleEvidence_FullVisibilityUnproven'
    else:
        visibility = 'UnknownOccludedOrMissing' if projected else 'OutsideAllViewsOrInvalidProjection'
    reasons = []
    if geometry != 'Valid': reasons.append('Geometry' + geometry)
    if cls != 'Valid': reasons.append('ClassUnknown')
    if not supported: reasons.append('NoSupportedVisibleAssociation')
    if conflicts: reasons.append('RenderIdentityConflict')
    if not obj.get('yaw_only_comparable', False): reasons.append('YawOnlyTiltUnsupported')
    if obj.get('reason') == 'HiddenBelowGround_NoTrainingTarget': reasons.append('UndergroundOutOfDomain')
    center = obj.get('center_hs_m')
    inrange = bool(center is not None and all(lo < x < hi for x, lo, hi in zip(center, [-51.2,-51.2,-5],[51.2,51.2,3])))
    crop = any(v['projection']['bbox_xyxy'][3]*.88 > 280 and v['projection']['bbox_xyxy'][1]*.88 < 792 for v in supported)
    if not inrange: reasons.append('OutsidePETRRangeOrUnknown')
    if not crop: reasons.append('NoSupportedViewInActualCrop')
    return dict(geometry_validity=geometry, geometry_basis='BoxAlgebra_NotMeshOrPhysicalContactCertification',
        class_validity=cls, class_basis='SourceTaxonomyAndAvailableSemanticEvidence_NotHumanSignoff',
        identity_validity=identity, visibility=visibility, human_review='Pending',
        supervision_eligibility=dict(positive=False, negative=False, ignore=True,
            provisional_positive_candidate=not reasons, reasons=reasons+['HumanSignoffMissing']),
        geometry_invalid_due_to_identity=False, old_label_status=obj['label_status'],
        old_positive_candidate=obj['positive_supervision_candidate'],
        PETR_common_region=inrange and crop)


def validate_signoff(candidate, entry):
    """Validate a reviewer-supplied artifact, not an identity authentication service.

    No default approval. Caller must retain the actual human-supplied source.
    A syntactically valid JSON alone does not prove a human performed the review.
    """
    required = {'candidate_id','candidate_sha256','decision','reviewer','reviewed_at',
                'review_source','geometry_confirmed','class_confirmed','identity_confirmed',
                'visible_confirmed','notes'}
    if set(entry) != required:
        raise ValueError('Review field mismatch')
    actual = digest({k:v for k,v in candidate.items() if k!='candidate_sha256'})
    if candidate['candidate_sha256'] != actual:
        raise ValueError('Candidate content changed since review material creation')
    if entry['candidate_id'] != candidate['candidate_id'] or entry['candidate_sha256'] != candidate['candidate_sha256']:
        raise ValueError('Review is not bound to this candidate')
    if entry['decision'] not in ('ApprovePositive','Reject','NeedsReview'):
        raise ValueError('Invalid review decision')
    for k in ('reviewer','reviewed_at','review_source'):
        if not isinstance(entry[k], str) or not entry[k].strip() or entry[k] in ('Pending','NotAvailable'):
            raise ValueError('Missing real review provenance')
    from datetime import datetime
    if datetime.fromisoformat(entry['reviewed_at'].replace('Z','+00:00')).tzinfo is None:
        raise ValueError('Review needs timezone')
    for k in ('geometry_confirmed','class_confirmed','identity_confirmed','visible_confirmed'):
        if type(entry[k]) is not bool: raise ValueError('Confirmation must be boolean')
    if entry['decision'] == 'ApprovePositive':
        if not all(entry[k] for k in ('geometry_confirmed','class_confirmed','identity_confirmed','visible_confirmed')):
            raise ValueError('Incomplete positive signoff')
        axes = candidate['trust']
        if axes['geometry_validity'] != 'Valid' or axes['class_validity'] != 'Valid':
            raise ValueError('Invalid geometry/class cannot be overridden by review')
        if not candidate['object'].get('yaw_only_comparable') or not axes['PETR_common_region']:
            raise ValueError('Model representation/range invalid')
        if candidate['object']['reason'] == 'HiddenBelowGround_NoTrainingTarget':
            raise ValueError('Underground target remains excluded')
    return entry['decision'] == 'ApprovePositive'


def hs_to_petr_center(box):
    """HS center,l,w,h,CCW yaw -> old PETR gravity center,w,l,h,CW yaw,0,0.

    Zero velocities are placeholders with mandatory zero velocity loss weights.
    """
    c = box['center_hs_m']; l,w,h = box['dimensions_lwh_m']; yaw = box['yaw_ccw_rad']
    values = [*c,w,l,h,(-yaw-math.pi/2+math.pi)%(2*math.pi)-math.pi,0.,0.]
    if not all(math.isfinite(x) for x in values) or min(l,w,h) <= 0:
        raise ValueError('Invalid target geometry')
    return values
