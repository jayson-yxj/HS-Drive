from types import SimpleNamespace as NS
import math
import pytest
from simulator.m6f_telemetry import scalar, serialize, observe, validate_pair, VEHICLE_FIELDS, WHEEL_FIELDS


def obj():
    return NS(**{k:0. for k in VEHICLE_FIELDS},wheels=[NS(**{k:1. for k in WHEEL_FIELDS}) for _ in range(4)])


def test_real_fields_preserved_and_not_mislabeled():
    t=obj();t.wheels[0].torque=-12.5;t.wheels[1].omega=3.
    r=serialize(t)
    assert r['wheels'][0]['torque']==-12.5
    assert r['wheels'][1]['omega']==3.
    assert 'brake_torque' not in r['wheels'][0]
    assert 'frame' not in r


@pytest.mark.parametrize('x',[math.nan,math.inf,-math.inf])
def test_engine_invalid_values_not_silently_zero(x):
    assert scalar(x)=={'invalid_raw':repr(x)}


def test_absent_engine_field_fails_explicitly():
    t=obj();del t.wheels[0].omega
    with pytest.raises(AttributeError):serialize(t)


@pytest.mark.parametrize('frames,valid',[([1,1],True),([1,2],False)])
def test_exclusive_frame_bracket(frames,valid):
    it=iter(frames);w=NS(get_snapshot=lambda:NS(frame=next(it),timestamp=NS(elapsed_seconds=.05)))
    r=observe(w,NS(get_telemetry_data=obj))
    assert (r['associated_frame'] is not None)==valid
    assert r['engine_telemetry_timestamp'] is None
    assert r['engine_substep'] is None


def test_frame_gap_rejected():
    with pytest.raises(ValueError):validate_pair({'associated_frame':1},{'associated_frame':3},3)
    with pytest.raises(ValueError):validate_pair({'associated_frame':1},{'associated_frame':None},2)
    validate_pair({'associated_frame':1},{'associated_frame':2},2)
