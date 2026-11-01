import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from datasets.m8j12_contract import CAMERAS
from simulator.m8j13_transport import Slots,SHAPE,SIZE


def payloads():return {n:bytes([i+1])*(SIZE//6) for i,n in enumerate(CAMERAS)}


def test_two_immutable_leases_full_reuse_and_stale_rejection(tmp_path):
    s=Slots(tmp_path/'slots',create=True);row={'frame':10,'timestamp':.5};identity={'run':'x','generation':2}
    first=s.publish('1',row,payloads(),identity);second=s.publish('2',row,payloads(),identity)
    assert s.publish('3',row,payloads(),identity) is None
    r,images,i=s.read(first);assert r==row and i==identity
    for k,n in enumerate(CAMERAS):assert images[n].shape==(396,704,3) and np.all(images[n]==k+1)
    s.release('1');third=s.publish('3',{'frame':11},payloads(),identity)
    with pytest.raises(RuntimeError,match='StaleLease'):s.read(first)
    assert s.read(second)[0]==row and s.read(third)[0]=={'frame':11}
    with pytest.raises(RuntimeError,match='InvalidLeaseRelease'):s.release('1')
    s.close();assert not s.root.exists()


def test_transport_rejects_changed_metadata_layout_and_short_payload(tmp_path):
    s=Slots(tmp_path/'slots',create=True)
    bad=payloads();bad[CAMERAS[0]]=b''
    with pytest.raises(ValueError,match='TransportSize'):s.publish('bad',{},bad,{})
    token=s.publish('a',{'frame':1},payloads(),{})
    p=s.root/'0.json';original=json.loads(p.read_text());changed=dict(original);changed['row']={'frame':2};p.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError,match='Binding'):s.read(token)
    changed=dict(original);changed['layout']='RGB';p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Layout'):s.read(token)
    s.close()


def test_reader_exit_leaves_creator_responsible_for_cleanup(tmp_path):
    s=Slots(tmp_path/'slots',create=True);token=s.publish('a',{'frame':1},payloads(),{})
    code="import json,sys,os; from simulator.m8j13_transport import Slots; r,im,i=Slots(sys.argv[1]).read(json.loads(sys.argv[2])); assert r['frame']==1; os._exit(17)"
    result=subprocess.run([sys.executable,'-c',code,str(s.root),json.dumps(token)])
    assert result.returncode==17 and s.read(token)[0]['frame']==1
    s.close();assert not s.root.exists()


def test_opt_in_budget_does_not_change_legacy_or_clocks():
    c=json.loads(Path('configs/m8j13_input_pipeline.json').read_text());old=json.loads(Path('configs/m8j13_parked_shadow.json').read_text())
    assert c['max_starts']==1 and c['max_forward_attempts']==32 and c['offline_forward_cap']==8 and c['total_forward_cap']==40
    assert sum(c[k] for k in ['settings_tick_charge','initialization_ticks','candidate_window_ticks','pending_drain_max_ticks','exit_observation_ticks'])<=800
    assert c['candidate_count']==120 and c['candidate_window_ticks']==480
    for k in ['input_wait_wall_s','max_plan_age_sim_s','max_plan_age_wall_s','max_command_age_wall_s','physics_dt_s','precision','server_root']:
        assert c[k]==old[k]


def test_only_one_preparing_or_waiting_item_and_bounded_close():
    import threading
    from simulator.m8j13_prepared_slot import PreparedSlot
    q=PreparedSlot();assert q.reserve();assert not q.reserve()
    assert q.publish('first');assert not q.reserve()
    assert q.take()=='first';assert q.reserve()
    assert q.publish('second')
    result=[];t=threading.Thread(target=lambda:result.append(q.publish('impossible-third')));t.start()
    q.close();t.join(1);assert not t.is_alive() and result==[False]


def test_old_wall_age_is_not_refreshed_by_preparation_or_waiting():
    from simulator.m8j13_safety import PlanStamp,freshness
    p=PlanStamp('run',10,1.,10.,10.1,10.29,'frozen')
    assert freshness(p,'run',14,1.2,10.29) is None
    assert freshness(p,'run',14,1.2,10.31)=='PlanWallExpired'
    assert freshness(p,'run',17,1.35,10.29)=='PlanSimulationExpired'
