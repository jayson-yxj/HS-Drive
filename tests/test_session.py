from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from simulator.session import CarlaSession


class FakeWorld:
    def __init__(self, fail=False):
        self.fail=fail
        self.applied=[]
    def get_settings(self):
        return SimpleNamespace(synchronous_mode=False,fixed_delta_seconds=None,no_rendering_mode=False)
    def get_map(self):
        return SimpleNamespace(name='Town03')
    def apply_settings(self,s):
        self.applied.append(s)
        if self.fail and len(self.applied)==1:
            raise RuntimeError('apply failure')


@pytest.mark.parametrize('fail',[False,True])
def test_settings_restored_after_exception(monkeypatch,fail):
    carla=pytest.importorskip('carla')
    world=FakeWorld(fail)
    client=Mock()
    client.get_client_version.return_value='0.9.15'
    client.get_server_version.return_value='0.9.15'
    client.get_world.return_value=world
    monkeypatch.setattr(carla,'Client',lambda *_:client)
    cfg={'host':'127.0.0.1','port':2000,'client_timeout_s':1,'map':'Town03','fixed_delta_seconds':.05}
    with pytest.raises(RuntimeError):
        with CarlaSession(cfg):
            raise RuntimeError('body failure')
    assert world.applied[0].synchronous_mode is True
    assert world.applied[-1].synchronous_mode is False
    assert world.applied[-1].fixed_delta_seconds is None


def test_cleanup_continues_after_sensor_failure():
    session=CarlaSession({})
    bad=Mock()
    bad.stop.side_effect=RuntimeError('dead sensor')
    good=Mock()
    session.sensors=[bad,good]
    session.actors=[bad,good]
    session.world=Mock()
    session.original=SimpleNamespace(synchronous_mode=False)
    with pytest.warns(UserWarning,match='sensor.stop failed'):
        session.__exit__(None,None,None)
    good.stop.assert_called_once()
    good.destroy.assert_called_once()
    bad.destroy.assert_called_once()
    session.world.apply_settings.assert_called_once_with(session.original)
