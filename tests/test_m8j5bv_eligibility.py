"""Synthetic eligibility regressions; no private payload and no torch required."""
import copy
import pytest

from datasets.m8j5b_coverage import positive_eligibility


def source():
    obj = dict(actor_id='unconfirmed-id', source_class='vehicle', raw={'base_type': 'car'},
        converted={'geometry_status': 'Valid', 'center_hs_m': [20., 0., 0.]},
        legacy={'geometry_status': 'Valid', 'yaw_only_comparable': True,
                'label_status': 'Unknown', 'reason': 'IdentityConflict'},
        views={'C0': {'raw_projection': {'status': 'Valid',
               'bbox_xyxy': [400, 400, 500, 500], 'near_plane_clipped': False}}})
    region = dict(camera='C0', semantic=14, pixels=100, semantic_agreement=1.,
        bbox_xyxy=[400, 400, 500, 500],
        raw_box_overlap_candidates=[{'actor_id': 'unconfirmed-id', 'pixels': 100, 'fraction': 1.}])
    return obj, region


@pytest.mark.parametrize('category', ['vehicle', 'walker'])
@pytest.mark.parametrize('distance', [5., 20., 45., 51.199])
def test_identity_unknown_does_not_invalidate_supported_source_box(category, distance):
    obj, region = source()
    obj['source_class'] = category
    obj['converted']['center_hs_m'][0] = distance
    if category == 'walker':
        region['semantic'] = 12
    result = positive_eligibility(obj, [region], ['C0'])
    assert result['eligible']
    assert result['identity_validity'] == 'NotRequiredForSourceBoxGeometry'
    assert not result['golden_gt']


@pytest.mark.parametrize('case,reason', [
    ('boundary', 'OutsideModelRange'), ('occluded', 'VisibleSemanticSupportUnconfirmedInCrop'),
    ('tiny', 'VisibleSemanticSupportUnconfirmedInCrop'),
    ('near_plane', 'VisibleSemanticSupportUnconfirmedInCrop'),
    ('invalid_geometry', 'GeometryUnreliable'),
    ('underground', 'UndergroundSourceRecord'),
])
def test_unreliable_support_not_promoted_to_positive(case, reason):
    obj, region = source()
    regions = [region]
    if case == 'boundary': obj['converted']['center_hs_m'][0] = 51.2
    elif case == 'occluded': regions = []
    elif case == 'tiny': region['pixels'] = 11
    elif case == 'near_plane': obj['views']['C0']['raw_projection']['near_plane_clipped'] = True
    elif case == 'invalid_geometry': obj['converted']['geometry_status'] = 'Invalid'
    elif case == 'underground': obj['legacy']['reason'] = 'HiddenBelowGround_NoTrainingTarget'
    result = positive_eligibility(obj, regions, ['C0'])
    assert not result['eligible'] and reason in result['reasons']


def test_two_camera_support_stays_one_source_object():
    obj, region = source()
    obj['views']['C1'] = copy.deepcopy(obj['views']['C0'])
    second = dict(region, camera='C1')
    result = positive_eligibility(obj, [region, second], ['C0', 'C1'])
    assert result['eligible'] and result['supported_cameras'] == ['C0', 'C1']
    assert not result['golden_gt']
