"""Opt-in runner intervention: official cold initialization, with read-only state audit.

Nothing is installed in the historical runner or official model by default. This
adapter only resets for cold/midpoint policies and observes original pre-update.
"""
import numpy as np
import torch

MEMORY_NAMES = ('memory_embedding', 'memory_reference_point', 'memory_timestamp',
                'memory_egopose', 'memory_velo')


def reset_official(model):
    model.prev_scene_token = None
    model.pts_bbox_head.reset_memory()
    assert all(getattr(model.pts_bbox_head, n) is None for n in MEMORY_NAMES)


class HistoryIntervention:
    def __init__(self, model, mode='normal', atol=1e-5, rtol=1e-5):
        if mode not in ('normal', 'cold', 'midpoint'):
            raise ValueError('UnknownHistoryMode')
        self.model=model; self.head=model.pts_bbox_head; self.mode=mode
        self.atol=atol; self.rtol=rtol; self.previous=None; self.history=[]
        self.row=None; self.entered=False; self.audit=None
        self.original_pre=self.head.pre_update_memory
        self.head.pre_update_memory=self._pre_update

    def close(self):
        self.head.pre_update_memory=self.original_pre

    def before(self, row):
        assert self.row is None, 'PreviousForwardNotCompleted'
        if self.previous and self.previous['scene_token']==row['scene_token']:
            assert row['timestamp_us']>self.previous['timestamp_us'], 'NonIncreasingTimestamp'
        scene_start=self.previous is None or self.previous['scene_token']!=row['scene_token']
        reset=(self.mode=='cold' or (self.mode=='midpoint' and row['scene_index']==row['reset_scene_index']))
        if scene_start or reset:
            self.history=[]
        if reset:
            reset_official(self.model)
        self.row=row; self.entered=False
        self.expected_prev=0 if scene_start or reset else 1
        self.audit=dict(reset_reason=row['reset_reason'],expected_prev_exists=self.expected_prev,
            detector_scene_before=self.model.prev_scene_token,mode=self.mode,
            input_dt_s=row['input_dt_s'],history_before=[dict(h) for h in self.history])

    def _pre_update(self, data):
        assert self.row is not None and not self.entered
        self.entered=True
        prev=float(data['prev_exists'].item())
        assert prev==self.expected_prev, 'HeadPrevExistsOverrideNotReset'
        current=float(data['timestamp'].item())
        assert current==self.row['timestamp_us']/1e6, 'InputTimestampChanged'
        assert self.model.prev_scene_token==self.row['scene_token'], 'DetectorSceneIdentityChanged'
        none={n:getattr(self.head,n) is None for n in MEMORY_NAMES}
        if not prev:
            assert all(none.values()), 'PartialReset'
            assert not self.history
        else:
            assert not any(none.values()) and self.history
        for h in self.history:
            assert h['scene_token']==self.row['scene_token']
            assert h['timestamp_us']<self.row['timestamp_us'], 'FutureMemory'
        self.audit.update(actual_head_prev_exists=prev,entry_memory_none=none,
                          detector_scene_at_head=self.model.prev_scene_token,
                          test_flag=bool(self.model.test_flag),single_test=bool(self.model.single_test))
        result=self.original_pre(data)
        # Official pre-update converts negative source seconds into positive relative ages.
        # Zero padding/pseudo-reference queries are legal cold initialization, not observations.
        n=min(self.head.memory_len,len(self.history)*self.head.topk_proposals)
        expected=[]
        for h in self.history:
            expected += [current-h['timestamp_us']/1e6]*self.head.topk_proposals
        actual=self.head.memory_timestamp[0,:n,0].detach().cpu().numpy()
        np.testing.assert_allclose(actual,np.asarray(expected[:n]),atol=self.atol,rtol=self.rtol)
        assert (actual>=0).all()
        self.audit.update(observed_slots_read=n,padded_slots=self.head.memory_len-n,
            age_seconds_min=float(actual.min()) if n else None,
            age_seconds_max=float(actual.max()) if n else None,
            age_max_abs_error=float(np.max(abs(actual-np.asarray(expected[:n])))) if n else 0.,
            cold_pseudo_reference_count=self.head.num_propagated if not prev else 0)
        for name in MEMORY_NAMES:
            assert torch.isfinite(getattr(self.head,name)).all(), 'NonfiniteMemory:'+name
        return result

    def after(self):
        assert self.entered, 'HeadInterventionNotObserved'
        f=self.row; current=f['timestamp_us']/1e6
        timestamps=self.head.memory_timestamp[0,:self.head.topk_proposals,0].detach().cpu().numpy()
        np.testing.assert_allclose(timestamps,-current,atol=self.atol,rtol=0.)
        assert all(torch.isfinite(getattr(self.head,n)).all() for n in MEMORY_NAMES)
        self.history.insert(0,{k:f[k] for k in ('index','sample_token','scene_token','timestamp_us')})
        max_blocks=(self.head.memory_len+self.head.topk_proposals-1)//self.head.topk_proposals+1
        self.history=self.history[:max_blocks]
        self.audit.update(write_frame=f['index'],write_timestamp_us=f['timestamp_us'],
            memory_after={n:dict(shape=list(getattr(self.head,n).shape),dtype=str(getattr(self.head,n).dtype)) for n in MEMORY_NAMES},
            history_after=[dict(h) for h in self.history])
        result=self.audit; self.previous=f; self.row=None
        return result
