import json
from pathlib import Path

import numpy as np
import pytest

from datasets.m8j5a_annotation_audit import SIGNS,converted_geometry,raw_corners,source_geometry_residuals,project_both
from datasets.m8j4_gt import transform


def sample(static=False):
    theta=.4;T=np.eye(4);T[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]];T[:3,3]=[10,2,1]
    local=np.array([.2,-.1,.8]);extent=np.array([2,1,.7])
    return dict(id='fixture',**{'class':'vehicle'},state='static' if static else 'dynamic',
        location=[999,999,999] if static else [10,2,1],rotation=[99,99,99] if static else [0,0,theta],
        bbx_loc=local.tolist(),extent=extent.tolist(),center=transform(local[None],T)[0].tolist(),
        world_cord=transform(SIGNS*extent+local,T).tolist(),world2vehicle=np.linalg.inv(T).tolist())


def test_extent_halfsize_center_not_actor_origin():
    o=sample();c=converted_geometry(o,np.eye(4))
    assert c['geometry_status']=='Valid' and np.allclose(c['dimensions_lwh_m'],[4,2,1.4])
    assert not np.allclose(c['center_hs_m'],o['location'])
    assert max(v for k,v in source_geometry_residuals(o).items() if isinstance(v,float))<1e-12


def test_static_ignores_buggy_actor_location_rotation_and_no_base_type():
    o=sample(True);a=converted_geometry(o,np.eye(4));o['location']=[-1000,0,0];o['rotation']=[0,0,0]
    assert a==converted_geometry(o,np.eye(4))
    assert a['geometry_status']=='Valid' and not a['training_authorized']
    expected=np.asarray(o.pop('world_cord'));reconstructed,basis=raw_corners(o)
    assert np.allclose(reconstructed,expected) and basis=='ActorLocalBoxInverseWorldTransform'


def test_missing_static_local_reference_is_not_guessed():
    o=sample(True);o.pop('world_cord');o.pop('bbx_loc')
    assert converted_geometry(o,np.eye(4))['geometry_status']=='Unknown'


def test_unknown_identity_or_class_does_not_delete_geometry():
    o=sample();o['base_type']='unmapped';o['identity_status']='Unknown'
    c=converted_geometry(o,np.eye(4));assert c['geometry_status']=='Valid'
    assert c['raw_record_preserved'] and not c['background_negative']


def test_center_or_extent_inconsistent_rejected():
    o=sample();o['center'][0]+=1
    assert converted_geometry(o,np.eye(4))['geometry_status']=='Unknown'
    o=sample();o['extent'][0]*=2
    assert converted_geometry(o,np.eye(4))['geometry_status']=='Unknown'


def test_world_ego_camera_projection_equivalence():
    o=sample();ego=np.eye(4);ego[:3,3]=[-2,1,0]
    sensor=dict(world2cam=np.eye(4),intrinsic=[[1000,0,800],[0,1000,450],[0,0,1]])
    a,b,error=project_both(o,converted_geometry(o,ego),sensor,ego)
    assert a['status']=='Valid' and error<1e-3


def test_level_box_proxy_is_not_training_label_or_trigger():
    o=dict(id='light',**{'class':'traffic_light'},center=[10,2,3],extent=[.1,.2,.6],trigger_volume_location=[100,100,100])
    c=converted_geometry(o,np.eye(4));assert c['geometry_status']=='Valid'
    assert 'Proxy' in c['reference'] and not c['training_authorized'] and c['source_class_mapping']=='NotPETRDetectionClass'
    assert np.allclose(c['center_hs_m'],[10,-2,3])


def test_all_actual_frames_lossless_chain_and_new_validation():
    from tools.m8j5a_audit import OUT
    files=sorted((OUT/'frames').glob('*.json'))
    if not files:pytest.skip('private payload unavailable')
    count=0;vehicles=0;walkers=0
    for p in files:
        f=json.loads(p.read_text());ego=next(o['raw'] for o in f['objects'] if o['source_class']=='ego_vehicle')
        for row in f['objects']:
            if row['source_class']=='ego_vehicle':continue
            count+=1;assert row['lineage']['legacy_sidecar'] and not row['lineage']['final_training_gt']
            g=converted_geometry(row['raw'],ego['world2ego'])
            assert g['geometry_status']=='Valid'
            if row['source_class'] in ('vehicle','walker'):
                assert row['legacy']['geometry_status']=='Valid'
                assert np.allclose(g['corners_hs_m'],row['legacy']['corners_hs_m'],atol=1e-6)
                vehicles+=row['source_class']=='vehicle';walkers+=row['source_class']=='walker'
    assert len(files)==214 and (count,vehicles,walkers)==(3741,1152,124)


def test_reproduction_00017_and_missing_walker_00124():
    from tools.m8j5a_audit import OUT
    if not (OUT/'frames/00017.json').exists():pytest.skip('private payload unavailable')
    f=json.loads((OUT/'frames/00017.json').read_text());m={o['actor_id']:o for o in f['objects']}
    assert m['393']['views']['CAM_BACK']['raw_projection']['status']=='Valid'
    assert m['394']['views']['CAM_FRONT']['converted_projection']['status']=='Valid'
    assert m['393']['lineage']['legacy_sidecar'] and m['394']['lineage']['legacy_sidecar']
    f=json.loads((OUT/'frames/00124.json').read_text())
    assert not any(o['source_class']=='walker' for o in f['objects'])
    assert any(r['semantic']==12 and r['pixels']>20000 and r['best_raw_overlap_fraction']==0 for r in f['render_regions'])
