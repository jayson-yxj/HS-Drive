"""Owned Traffic Manager setup/cleanup failure paths without running a simulator."""
import sys,types
import pytest
from simulator.session import CarlaSession


def test_tm_lifecycle_and_busy_port(monkeypatch):
    import socket
    class Probe:
        busy=False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def connect_ex(self,address):return 0 if self.busy else 111
    monkeypatch.setattr(socket,'socket',Probe)
    monkeypatch.setitem(sys.modules,'carla',types.SimpleNamespace(VehicleControl=lambda:None))
    calls=[]
    class TM:
        def set_synchronous_mode(self,x):calls.append(('sync',x))
        def set_random_device_seed(self,x):calls.append(('seed',x))
        def auto_lane_change(self,ego,x):calls.append(('lane_change',x))
        def set_desired_speed(self,ego,x):calls.append(('kmh',x))
        def get_port(self):return 8050
    class Ego:
        def apply_control(self,x):pass
        def set_autopilot(self,x,p):calls.append(('autopilot',x,p))
    session=CarlaSession({'host':'127.0.0.1','seed':17})
    session.client=types.SimpleNamespace(get_trafficmanager=lambda p:TM())
    Probe.busy=True
    with pytest.raises(RuntimeError,match='already in use'):session.start_autopilot(Ego(),port=8050,speed_mps=5)
    assert session.traffic_manager is None
    Probe.busy=False;session.start_autopilot(Ego(),port=8050,speed_mps=5);session.__exit__()
    assert calls==[('sync',True),('seed',17),('lane_change',False),('kmh',18.),('autopilot',True,8050),('autopilot',False,8050),('sync',False)]


def test_tm_sync_cleanup_survives_autopilot_failure():
    from unittest.mock import Mock
    session=CarlaSession({});session.traffic_manager=Mock();session.autopilot_ego=Mock()
    session.autopilot_ego.set_autopilot.side_effect=RuntimeError('dead vehicle')
    with pytest.warns(UserWarning,match='Autopilot cleanup'):session.__exit__()
    session.traffic_manager.set_synchronous_mode.assert_called_once_with(False)
