import json
from pathlib import Path
import numpy as np
import pytest

def test_inference_never_loads_evidence_sidecar(monkeypatch):
    pytest.importorskip('torch')
    from datasets.m48_training import EvidenceDataset
    root=Path('/root/gpufree-data/hs-drive-data/processed/m46_training_dense_index')
    if not root.exists():pytest.skip('server data not installed')
    def fail(*a,**k):raise AssertionError('GT read during inference')
    monkeypatch.setattr(np,'load',fail)
    ds=EvidenceDataset(root,'train','does_not_exist',supervision=False)
    item=ds[0];assert item.targets is None
    with pytest.raises(KeyError):item.inputs['surface_evidence']


def test_road_no_fast_acceptance_without_explicit_optin():
    from simulator.m48_road import CertifiedRoad
    class Map:
        def to_opendrive(self):return '<OpenDRIVE/>'
    road=CertifiedRoad(Map());calls=[]
    road.exact=lambda x,y: calls.append((x,y)) or x<1
    out=road.batch(np.array([[0,0],[2,0]]))
    assert out.tolist()==[True,False] and len(calls)==2
    assert not road.certified([[0,0]]).any()


def test_straight_certificate_rejects_holes_curves_and_boundaries():
    from simulator.m48_road import CertifiedRoad
    class Map:
        def to_opendrive(self):
            return '''<OpenDRIVE><road length="10" junction="-1"><planView><geometry s="0" x="0" y="0" hdg="0" length="10"><line/></geometry></planView><lanes><laneSection s="0"><right><lane id="-1" type="driving"><width sOffset="0" a="4" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>'''
    road=CertifiedRoad(Map());assert road.certified([[5,-2],[5,0],[5,-4],[-1,-2],[12,-2]]).tolist()==[True,False,False,False,False]
    class Curved(Map):
        def to_opendrive(self):return super().to_opendrive().replace('<line/>','<arc curvature=".1"/>')
    assert not CertifiedRoad(Curved()).certified([[5,-2]]).any()
    class Junction(Map):
        def to_opendrive(self):return super().to_opendrive().replace('junction="-1"','junction="10"')
    assert not CertifiedRoad(Junction()).certified([[5,-2]]).any()


def test_multirate_interpolates_world_coordinates_and_expires_without_refresh():
    from simulator.m48_schedule import TimedPlan
    from simulator.oracle_driver import OracleState,Trajectory
    queue=TimedPlan();s=OracleState(10,20,0,1,1.)
    t=np.arange(1,61)*.1;points=np.c_[t,np.zeros(60),np.zeros(60),np.ones(60)]
    queue.store(Trajectory(points,1.,.1),s)
    later=OracleState(10.15,20,0,1,1.15);p,reason=queue.sample(later)
    assert reason is None and abs(p.points[0,0]-.1)<1e-6
    assert not queue.due(1.15) and queue.due(1.2)
    later.timestamp_s=1.3
    assert queue.sample(later)[1]=='trajectory_timeout'
    assert queue.plan.timestamp_s==1.


def test_scheduled_generation_snapshot_is_immutable_to_caller_mutation():
    from simulator.m48_schedule import TimedPlan
    from simulator.oracle_driver import OracleState,Trajectory
    state=OracleState(0,0,0,1,1.);points=np.c_[np.arange(1,61)*.1,np.zeros((60,2)),np.ones(60)]
    tr=Trajectory(points,1.,.1);schedule=TimedPlan();schedule.store(tr,state)
    tr.timestamp_s=-20;tr.points[:]=np.nan;state.x=100
    actual=OracleState(.1,0,0,1,1.1);p,reason=schedule.sample(actual)
    assert schedule.plan.timestamp_s==1 and schedule.state.x==0
    assert reason is None and np.isfinite(p.points).all()
    assert abs(p.points[0,0]-.1)<1e-6
