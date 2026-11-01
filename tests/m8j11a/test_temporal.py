from types import SimpleNamespace as S
import numpy as np
import pytest
import torch
from datasets.m8j11a_temporal import HistoryIntervention,MEMORY_NAMES
from tools.m8j11a_common import schedules,compare_arrays
from tools.m8j11a_budget import reserve

class Head:
    memory_len=4;topk_proposals=2;num_propagated=2
    def __init__(self):self.reset_memory()
    def reset_memory(self):
        for n in MEMORY_NAMES:setattr(self,n,None)
    def pre_update_memory(self,data):
        if self.memory_embedding is None:
            for n in MEMORY_NAMES:setattr(self,n,torch.zeros(1,4,1,dtype=torch.float64))
            self.memory_reference_point[:,:2]=3 # legal pseudo reference, not stale memory
        else:
            self.memory_timestamp+=data['timestamp'].view(1,1,1)
            for n in MEMORY_NAMES:setattr(self,n,getattr(self,n)[:,:4]*data['prev_exists'].view(1,1,1))
    def post(self,t):
        for n in MEMORY_NAMES:setattr(self,n,torch.cat([torch.zeros(1,2,1,dtype=torch.float64),getattr(self,n)],1))
        self.memory_timestamp-=t

def model():return S(prev_scene_token=None,pts_bbox_head=Head(),test_flag=False,single_test=False)
def row(i,scene='s'):return dict(index=i,scene_index=i,reset_scene_index=2,scene_token=scene,sample_token=scene+str(i),timestamp_us=10000000+i*500000,reset_reason=None,input_dt_s=None if i==0 else .5)
def forward(m,a,r,break_prev=False):
    a.before(r)
    new=m.prev_scene_token!=r['scene_token'];m.prev_scene_token=r['scene_token']
    if new:m.pts_bbox_head.reset_memory()
    prev=1. if break_prev else float(not new)
    d=dict(prev_exists=torch.tensor([prev]),timestamp=torch.tensor([r['timestamp_us']/1e6],dtype=torch.float64))
    m.pts_bbox_head.pre_update_memory(d);m.pts_bbox_head.post(d['timestamp'].item())
    return a.after()

@pytest.mark.parametrize('mode,expected',[('normal',[0,1,1,1]),('cold',[0,0,0,0]),('midpoint',[0,1,0,1])])
def test_actual_head_flags_and_ages(mode,expected):
    m=model();a=HistoryIntervention(m,mode)
    got=[forward(m,a,row(i)) for i in range(4)]
    assert [r['actual_head_prev_exists'] for r in got]==expected
    assert all(r['age_max_abs_error']==0 for r in got)
    assert got[0]['cold_pseudo_reference_count']==2
    a.close()

def test_default_normal_scene_boundary_and_sparse_real_time():
    m=model();a=HistoryIntervention(m)
    forward(m,a,row(0));d=forward(m,a,row(2));assert d['age_seconds_max']==1.
    d=forward(m,a,row(0,'new'));assert d['observed_slots_read']==0 and d['actual_head_prev_exists']==0

def test_detect_upper_entry_ignoring_cold_reset():
    m=model();a=HistoryIntervention(m,'cold')
    with pytest.raises(AssertionError,match='HeadPrevExists'):forward(m,a,row(0),True)

def test_future_or_duplicate_time_rejected():
    m=model();a=HistoryIntervention(m);forward(m,a,row(2))
    with pytest.raises(AssertionError,match='NonIncreasing'):a.before(row(1))

def test_schedule_fixed_counts_and_resets():
    f=[dict(index=j+i,sample_token=s+str(i),scene_token=s,scene_name=s,timestamp_us=1000000+i*500000) for s,n,j in [('a',40,0),('b',41,40)] for i in range(n)]
    out,scenes=schedules(f)
    assert [len(out[x]) for x in ('T1','T2','T3')]==[81,81,41]
    assert [s['reset_scene_index'] for s in scenes]==[20,20]
    assert [r['index'] for r in out['T2'] if r['reset_reason']=='mid_scene_reset']==[20,60]
    assert all(r['input_dt_s'] in (None,1.) for r in out['T3'])

def test_budget_failed_reservation_counts_and_bound(tmp_path):
    p=tmp_path/'budget.jsonl';reserve(p,'r','T1',0,maximum=2);reserve(p,'r','T1',1,maximum=2)
    with pytest.raises(RuntimeError,match='ForwardBudget'):reserve(p,'r','T2',0,maximum=2)
    assert len(p.read_text().splitlines())==2

def test_output_comparison_no_reordering_or_tolerance_relaxation():
    a={'scores':np.array([.5,.6]),'labels':np.array([1,2])}
    assert compare_arrays(a,a)['scores']==0
    with pytest.raises(AssertionError):compare_arrays(a,dict(a,labels=np.array([2,1])))

def test_partial_memory_reset_rejected():
    m=model();a=HistoryIntervention(m,'cold');r=row(0);a.before(r)
    m.prev_scene_token=r['scene_token'];m.pts_bbox_head.memory_velo=torch.ones(1,4,1)
    with pytest.raises(AssertionError,match='PartialReset'):
        m.pts_bbox_head.pre_update_memory(dict(prev_exists=torch.tensor([0.]),timestamp=torch.tensor([10.],dtype=torch.float64)))

def test_wrong_memory_age_is_detected():
    m=model();a=HistoryIntervention(m);forward(m,a,row(0));m.pts_bbox_head.memory_timestamp[:,0]+=1
    with pytest.raises(AssertionError):forward(m,a,row(1))

def test_source_scene_and_time_not_fabricated_and_default_observer_same():
    reference=model();observed=model();adapter=HistoryIntervention(observed)
    for r in [row(0),row(1),row(2),row(0,'next')]:
        saved=dict(r)
        new=reference.prev_scene_token!=r['scene_token'];reference.prev_scene_token=r['scene_token']
        if new:reference.pts_bbox_head.reset_memory()
        t=r['timestamp_us']/1e6
        reference.pts_bbox_head.pre_update_memory(dict(prev_exists=torch.tensor([float(not new)]),timestamp=torch.tensor([t],dtype=torch.float64)))
        reference.pts_bbox_head.post(t);forward(observed,adapter,r)
        assert r==saved
        for n in MEMORY_NAMES:torch.testing.assert_close(getattr(reference.pts_bbox_head,n),getattr(observed.pts_bbox_head,n),atol=0,rtol=0)

def test_official_cold_initialization_without_full_forward():
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
    head=Head();head.embed_dims=3;head.pc_range=torch.tensor([-1.,-2.,-3.,1.,2.,3.])
    head.pseudo_reference_points=S(weight=torch.full((2,3),.75))
    m=S(prev_scene_token='old',pts_bbox_head=head,test_flag=False,single_test=False)
    head.pre_update_memory=lambda data:StreamPETRHead.pre_update_memory(head,data)
    adapter=HistoryIntervention(m,'cold');r=row(0);adapter.before(r)
    assert m.prev_scene_token is None
    m.prev_scene_token=r['scene_token']
    head.pre_update_memory(dict(prev_exists=torch.tensor([0.]),timestamp=torch.tensor([10.],dtype=torch.float64)))
    assert adapter.audit['actual_head_prev_exists']==0
    assert head.memory_embedding.count_nonzero()==0
    assert head.memory_reference_point[:,:2].count_nonzero()>0
    assert torch.equal(head.memory_egopose[0,0],torch.eye(4))

def test_budget_condition_cap(tmp_path):
    p=tmp_path/'ledger';reserve(p,'r','T1',0,maximum=243,primary_limit=1)
    with pytest.raises(RuntimeError,match='ConditionBudget'):reserve(p,'r','T1',1,maximum=243,primary_limit=1)
    reserve(p,'other','T2',0,maximum=243,primary_limit=1)
