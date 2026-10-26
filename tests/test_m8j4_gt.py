import json
from pathlib import Path
import numpy as np
import pytest
from datasets.m8j4_gt import box_to_hs,project_box,regions,associate,enforce_one_to_one

CFG=json.loads(Path('configs/m8j4_protocol.json').read_text())['association']


def obj():
    c=np.array([10.,2.,1.]); dims=np.array([4.,2.,2.])
    corners=np.array([[x,y,z] for x in [.5,-.5] for y in [.5,-.5] for z in [.5,-.5]])*dims+c
    return dict(id=3,**{'class':'vehicle'},type_id='vehicle.synthetic',base_type='car',
                center=c.tolist(),extent=(dims/2).tolist(),world_cord=corners.tolist())


def test_center_dims_yaw_and_corner_roundtrip():
    b=box_to_hs(obj(),np.eye(4))
    assert b['geometry_status']=='Valid'
    assert np.allclose(b['center_hs_m'],[10,-2,1])
    assert np.allclose(b['dimensions_lwh_m'],[4,2,2])
    assert b['yaw_ccw_rad']==0 and not b['training_negative']
    assert np.allclose(np.asarray(b['corners_hs_m'])*[1,-1,1],obj()['world_cord'])


def test_rotated_ego_yaw_and_invalid():
    T=np.eye(4);T[:2,:2]=[[0,1],[-1,0]]
    b=box_to_hs(obj(),T);assert np.isclose(b['yaw_ccw_rad'],np.pi/2)
    o=obj();o['extent'][0]=0
    assert box_to_hs(o,np.eye(4))['geometry_status']=='Invalid'
    o=obj();o['class']='traffic_light'
    assert box_to_hs(o,np.eye(4))['geometry_status']=='Unknown'


def test_near_plane_and_behind():
    k=np.array([[50,0,50],[0,50,50],[0,0,1.]])
    o=obj();p=project_box(o['world_cord'],np.eye(4),k,100,100)
    assert p['status']=='Valid' and not p['near_plane_clipped']
    c=np.array(o['world_cord']);c[:,0]-=10
    p=project_box(c,np.eye(4),k,100,100);assert p['near_plane_clipped']
    c[:,0]-=20;assert project_box(c,np.eye(4),k,100,100)['reason']=='BehindCamera'
    c=np.array(o['world_cord']);c[:,1]+=100
    assert project_box(c,np.eye(4),k,100,100)['status']=='Invalid'


def test_instance_token_not_actor_and_semantic_support():
    im=np.zeros((40,40,4),np.uint8);sem=np.zeros((40,40),np.uint8)
    im[10:30,10:30]=[122,233,12,255];sem[10:30,10:30]=12
    labels,token,rs=regions(im,sem)
    assert rs[0]['render_token']==31465 and rs[0]['render_token']!=358
    p=dict(status='Valid',polygon=[[10,10],[30,10],[30,30],[10,30]],
           bbox_xyxy=[10,10,30,30],near_plane_clipped=False)
    a=associate(p,'walker',labels,token,rs,CFG);assert a['status']=='Valid'
    views=enforce_one_to_one({'a':dict(a),'b':dict(a)})
    assert all(x['status']=='Unknown' for x in views.values())
    rs[0]['semantic_agreement']=.2
    assert associate(p,'walker',labels,token,rs,CFG)['status']=='Unknown'
    assert associate(p,'vehicle',labels,token,rs,CFG)['status']=='Unknown'


def test_same_region_under_occlusion_remains_uncertain():
    p=dict(status='Invalid',reason='BehindCamera')
    assert associate(p,'walker',None,None,[],CFG)['status']=='Invalid'
    p=dict(status='Valid',near_plane_clipped=True)
    assert associate(p,'walker',None,None,[],CFG)['status']=='Unknown'
