"""Fixed visible detection task, not an occupancy-free certificate."""
import numpy as np

from datasets.m8j5b_coverage import positive_eligibility


def task_eligibility(obj, regions, cameras, bounds):
    result=positive_eligibility(obj,regions,cameras)
    center=np.asarray(obj['converted'].get('center_hs_m',[np.nan]*3))
    in_range=bool(np.isfinite(center).all() and np.all(center>bounds[:3]) and np.all(center<bounds[3:]))
    reasons=list(result['reasons'])
    if not in_range:reasons.append('OutsidePredeclaredVisibleDetectionRange')
    return dict(result,eligible=not reasons,reasons=reasons,in_task_range=in_range,
        negative_authorized=False,scope='VisibleDetection_NotOccupancyFree',human_review='Pending')


def region_status(region):
    """Existing renderer-region evidence; token is not physical actor identity."""
    if region['semantic'] not in (12,14):
        return 'OutsideSelectedClass_StillPreserve'
    if region['best_raw_overlap_fraction']>=.5:
        return 'RawBoxSpatialSupport_NotIdentityOrVisibilityCertification'
    # Even a pedestrian overlapping ego projection cannot be automatically erased.
    return 'UnexplainedForeground_DepthAndTaskMembershipUnknown'


def coverage_qualification(confirmed_gaps, unexplained_regions, full_review, ignore_plan_verified=False):
    """Unknown can coexist with valid task negatives via a reviewed Ignore plan.

    It need not be reconstructed as a 3D box or proved physically empty.
    This flag requires actual coverage and exclusion-handling evidence; the
    current eight frames do not have it.
    """
    reasons=[]
    if confirmed_gaps and not ignore_plan_verified:reasons.append('ConfirmedSourceBoxGapRequiresVerifiedIgnore')
    if unexplained_regions and not ignore_plan_verified:reasons.append('UnexplainedForegroundTaskMembershipOrIgnoreUnresolved')
    if not full_review:reasons.append('AllSixViewTaskCoverageReviewPending')
    return dict(qualified=not reasons,reasons=reasons,
        semantics='ExhaustiveInScopeVisibleObjectAnnotations_NotPhysicalFree',
        ignore_plan_verified=ignore_plan_verified,
        negative_enabled=False)  # This diagnostic cannot grant training permission.
