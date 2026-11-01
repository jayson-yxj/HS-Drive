"""Actual owner-snapshot confirmation and unchanged dual-clock freshness.

A pending tick gives a simulation-age interval, not a confirmed current age.
Each call records its own wait; callers must sum all calls for an observation.
"""
import json,threading,time
from pathlib import Path
from simulator.m8j13_safety import PlanStamp,freshness
from simulator.m8j13_shadow_contract import snapshot_is_current
from tools.m8j13_common import append

class SnapshotAge:
    def __init__(self,root,require_confirmation=True,clock=time.monotonic,sleep=time.sleep):
        self.root=Path(root);self.require_confirmation=require_confirmation;self.clock=clock;self.sleep=sleep;self.lock=threading.Lock();self.records=[]
    def __call__(self,row,start,wait_for_snapshot=False,phase='unspecified'):
        begin=self.clock();until=min(begin+.06,row['input_capture_wall_s']+.3);wait_start=None;polls=0
        while True:
            pending_path=self.root/'tick_pending.json'
            first=json.loads(pending_path.read_text()) if self.require_confirmation and pending_path.exists() else None
            state=json.loads((self.root/'current_snapshot.json').read_text());now=self.clock()
            second=json.loads(pending_path.read_text()) if first is not None else None
            confirmed=not self.require_confirmation or snapshot_is_current(first,second,state)
            stamp=PlanStamp(row['sequence_id'],row['frame_id'],row['timestamp_s'],row['input_capture_wall_s'],start,now,'frozen')
            raw=freshness(stamp,row['sequence_id'],state['frame'],state['sim_s'],now)
            wall_age=now-row['input_capture_wall_s'];lower=state['sim_s']-row['timestamp_s'];upper=None
            if confirmed:upper=lower
            elif first==second and second is not None and second['pending'] and second['frame']>=state['frame']:
                upper=lower+(second['frame']-state['frame'])*.05
            wall_expired=wall_age>.3+1e-6;sim_expired=lower>.3+1e-6 if confirmed else None
            if raw in ('InvalidPlanClock','CrossSequencePlan','FuturePlan'):reason=raw
            elif not confirmed:reason='PlanWallExpired' if wall_expired else 'SimulationSnapshotUnconfirmed'
            else:reason=raw
            if reason=='SimulationSnapshotUnconfirmed' and wait_for_snapshot and now<until:
                if wait_start is None:wait_start=now
                polls+=1;self.sleep(.001);continue
            wait_ms=(now-wait_start)*1000 if wait_start is not None else 0.
            result=dict(reason=reason,wall_age_s=wall_age,sim_age_s=lower if confirmed else None,
                sim_age_range_s=[lower,upper],checked_wall_s=now,snapshot_confirmed=confirmed,
                wall_expired=wall_expired,simulation_expired=sim_expired,phase=phase,
                check_begin_wall_s=begin,check_end_wall_s=now,snapshot_wait_begin_wall_s=wait_start,
                snapshot_wait_end_wall_s=now if wait_start is not None else None,snapshot_wait_ms=wait_ms,polls=polls,
                sim_age_scope='Confirmed owner Snapshot or explicit range during in-flight tick; no freshness claim from unconfirmed state')
            record=dict(input_frame=row['frame_id'],input_wall_s=row['input_capture_wall_s'],**result)
            with self.lock:
                self.records.append(record);append(self.root/'snapshot_checks.jsonl',record)
            return result
    def cumulative_wait_ms(self,frame):
        with self.lock:return sum(x['snapshot_wait_ms'] for x in self.records if x['input_frame']==frame)


def ledger_entries(path):
    path=Path(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def consumed_completion(response,row,frame,sim_s,received_wall_s,checked_wall_s):
    """Same consumer used by owner and process-boundary preflight; no actuation."""
    from simulator.m8j13_shadow_contract import require_completion_only
    require_completion_only(response)
    stamp=PlanStamp(row['sequence_id'],row['frame_id'],row['timestamp_s'],row['input_capture_wall_s'],row['input_capture_wall_s'],checked_wall_s,'frozen')
    reason=freshness(stamp,row['sequence_id'],frame,sim_s,checked_wall_s)
    checks=bool(response.get('model_forward') and response.get('finite') and response.get('trajectory_valid',True))
    return dict(wall_age_s=checked_wall_s-row['input_capture_wall_s'],sim_age_s=sim_s-row['timestamp_s'],age_reason=reason,
        wall_expired=checked_wall_s-row['input_capture_wall_s']>.3+1e-6,simulation_expired=sim_s-row['timestamp_s']>.3+1e-6,
        snapshot_confirmed=True,necessary_checks_passed=checks,consumable_fresh=checks and reason is None,
        received_wall_s=received_wall_s,owner_consume_check_done_wall_s=checked_wall_s)
