from types import SimpleNamespace
import pytest
from simulator.m5e_affinity import AffinityPlanner
import simulator.m5e_affinity as module


@pytest.mark.parametrize('spec,expected',[('',[]),('2,3',[(123,{2,3}),(124,{2,3})])])
def test_affinity_opt_in_sets_all_worker_threads_only(monkeypatch,spec,expected):
    calls=[]
    monkeypatch.setenv('M5E_WORKER_CPUS',spec)
    monkeypatch.setattr(module.AsyncPlanner,'__init__',lambda self,*a,**kw:setattr(self,'worker',SimpleNamespace(pid=123)))
    monkeypatch.setattr(module,'Path',lambda path:SimpleNamespace(iterdir=lambda:[SimpleNamespace(name='123'),SimpleNamespace(name='124')]))
    monkeypatch.setattr(module.os,'sched_setaffinity',lambda pid,cpus:calls.append((pid,cpus)))
    AffinityPlanner('map','route')
    assert calls==expected
