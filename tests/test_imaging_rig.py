from types import SimpleNamespace
import pytest
from simulator.camera_rig import spawn_rig
from simulator.imaging import load_camera_config


def test_photometric_attributes_are_not_applied_to_depth():
    pytest.importorskip('carla')
    calls=[]
    class BP:
        def __init__(self,kind):self.kind=kind;self.attrs={}
        def has_attribute(self,key):return True
        def set_attribute(self,key,value):self.attrs[key]=value
    class Session:
        world=SimpleNamespace(get_blueprint_library=lambda:SimpleNamespace(find=lambda kind:BP(kind)))
        def spawn(self,bp,mount,parent):
            calls.append((bp.kind,dict(bp.attrs)))
            return SimpleNamespace(attributes=bp.attrs,listen=lambda callback:None)
    cfg=load_camera_config('configs/cameras.yaml','configs/imaging/training_960_v4.yaml')
    cfg['render_sensor_tick_s'] = 0.0
    spawn_rig(Session(),object(),cfg,SimpleNamespace(callback=lambda name:None),('front',))
    rgb,depth=calls
    assert rgb[1]['sensor_tick']=='0.0' and cfg['sensor_tick_s']==.1
    assert rgb[1]['enable_postprocess_effects']=='true'
    assert 'enable_postprocess_effects' not in depth[1] and 'exposure_compensation' not in depth[1]
    for key in ('image_size_x','image_size_y','fov','sensor_tick','lens_k','lens_kcube'):
        assert rgb[1][key]==depth[1][key]
