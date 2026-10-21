import pytest
from simulator.m5k_input import LatestRGBPacket

def test_timestamps_expiry_order_and_recovery():
    p=LatestRGBPacket();assert p.accept([10]*6,[1.]*6,1.1)==1
    for f,t,now in [([10]*6,[1.]*6,1.1),([11]*6,[2.]*6,2.251),([11]*6,[2.]*6,1.9),([11]*5+[12],[2.]*6,2.1),([11]*6,[2.]*5+[2.01],2.1),([11]*6,[float('nan')]*6,2.1)]:
        with pytest.raises(ValueError):p.accept(f,t,now)
    assert p.accept([12]*6,[3.]*6,3.1)==3
