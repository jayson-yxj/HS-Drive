"""Synthetic visible-detection scope tests; independent of private payload."""
import runpy
from pathlib import Path

import numpy as np
import pytest

from datasets.m8j5c_detection_scope import task_eligibility,region_status,coverage_qualification

source=runpy.run_path(str(Path(__file__).with_name('test_m8j5bv_eligibility.py')))['source']
BOUNDS=np.asarray([-40,-40,-3,40,40,3])


def test_identity_unknown_still_positive_diagnostic():
    obj,r=source()
    out=task_eligibility(obj,[r],['C0'],BOUNDS)
    assert out['eligible'] and not out['negative_authorized']
    assert out['identity_validity']=='NotRequiredForSourceBoxGeometry'


@pytest.mark.parametrize('center',[[40,0,0],[41,0,0],[0,40,0],[0,0,3]])
def test_scope_boundary_excluded_not_background(center):
    obj,r=source();obj['converted']['center_hs_m']=center
    out=task_eligibility(obj,[r],['C0'],BOUNDS)
    assert not out['eligible'] and not out['negative_authorized']
    assert 'OutsidePredeclaredVisibleDetectionRange' in out['reasons']


def test_invisible_source_not_erased_into_negative():
    obj,_=source();out=task_eligibility(obj,[],['C0'],BOUNDS)
    assert not out['eligible'] and 'VisibleSemanticSupportUnconfirmedInCrop' in out['reasons']
    assert not out['negative_authorized']


def test_tiny_unboxed_region_remains_unexplained():
    assert region_status(dict(semantic=14,best_raw_overlap_fraction=0,pixels=1)).startswith('Unexplained')


def test_ego_overlap_does_not_erase_pedestrian():
    assert region_status(dict(semantic=12,best_raw_overlap_fraction=0,ego_overlap_fraction=1)).startswith('Unexplained')


def test_source_spatial_support_not_identity():
    assert region_status(dict(semantic=14,best_raw_overlap_fraction=1)).endswith('NotIdentityOrVisibilityCertification')


@pytest.mark.parametrize('gap,unknown,review',[(1,0,True),(0,1,True),(0,0,False)])
def test_incomplete_task_coverage_cannot_qualify(gap,unknown,review):
    assert not coverage_qualification(gap,unknown,review)['qualified']


def test_qualified_task_definition_still_does_not_authorize_training():
    out=coverage_qualification(0,0,True)
    assert out['qualified'] and not out['negative_enabled']
    assert out['semantics'].endswith('NotPhysicalFree')


def test_smoke_preflight_does_not_promote_precision_to_supervision():
    import json
    from tools.m8j5c_smoke_preflight import preflight
    config=json.loads(Path('configs/m8j5c_smoke_proposal.json').read_text())
    result=preflight(config)
    assert result['status']=='NoGo' and result['optimizer_steps']==0 and not result['model_loaded']
    assert 'currently_missing_detection_coverage' in result['reasons']


def test_known_gap_can_coexist_with_reviewed_detection_ignore_contract():
    result=coverage_qualification(1,2,True,ignore_plan_verified=True)
    assert result['qualified'] and not result['negative_enabled']
    assert result['ignore_plan_verified']  # synthetic contract, never a real signoff
