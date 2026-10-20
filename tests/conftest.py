"""Small real-format six-camera recordings for M3 corruption/leakage tests."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from datasets.schema import SensorSample,FrameBundle,CAMERA_IDS
from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical
from simulator.recorder import Recorder


@pytest.fixture
def m3_registry(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    scenes=[]
    for n,(sid,split,town) in enumerate([('train_a','train','Town03'),('train_b','train','Town03'),
                                        ('val','val','Town03'),('test','test','Town10HD_Opt')]):
        mount=np.eye(4);pose=np.eye(4);pose[0,3]=n*200
        cal={'cameras':{c:{'width':8,'height':4,'K':intrinsics_from_hfov(8,4,100).tolist(),
            'T_carla_ego_from_camera':mount.tolist(),
            'T_hs_ego_from_opt':camera_mount_to_hs_from_optical(mount).tolist()} for c in CAMERA_IDS}}
        meta={'carla_version':'0.9.15','map':town,'seed':7,'weather':{'sun_altitude_angle':50},
              'camera_config':{'sensor_tick_s':.1,'imaging_profile_id':'fixture'}}
        recorder=Recorder(raw/sid,cal,meta)
        for i in range(4):
            frame=100+2*i;timestamp=10+i*.1
            sensors={}
            for j,c in enumerate(CAMERA_IDS):
                rgb=np.tile(np.array([20+n*30+i,60+j,140,255],np.uint8),(4,8,1))
                value=round(10/1000*(256**3-1))
                depth=np.tile(np.array([value>>16,(value>>8)&255,value&255,255],np.uint8),(4,8,1))
                for kind,pixels in [('rgb',rgb),('depth',depth)]:
                    name=f'{c}/{kind}'
                    sensors[name]=SensorSample(name,frame,timestamp,8,4,pixels.tobytes(),pose)
            ego={'T_world_from_ego':pose.tolist(),'T_carla_world_from_ego':pose.tolist(),
                 'speed_mps':0.,'velocity_world_mps':[0,0,0],'angular_velocity_world_radps':[0,0,0]}
            recorder.write(FrameBundle(frame,timestamp,sensors,ego,[{'actor_id':1,'is_ego':True}]))
        recorder.close('complete',{'complete':4})
        scenes.append({'scene_id':sid,'path':sid,'group_id':sid,'split':split})
    evidence=tmp_path/'depth.json'
    evidence.write_text(json.dumps({'convention':'optical_z','status':'verified_fixture'}))
    registry={'schema_version':'hsd-registry-v1','dataset_version':'fixture','raw_root':'raw',
        'depth_evidence':'depth.json','allowed_profiles':['fixture'],'scope':'synthetic test',
        'split_policy':{'seed':7,'val_fraction':.2,'test_maps':['Town10HD_Opt'],'min_cross_split_distance_m':100},
        'scenes':scenes}
    path=tmp_path/'registry.yaml';path.write_text(yaml.safe_dump(registry))
    return path


@pytest.fixture
def m3_index(m3_registry):
    from datasets.index import build_index
    output=m3_registry.parent/'index'
    build_index(m3_registry,output)
    return output
