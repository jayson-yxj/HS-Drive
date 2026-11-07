"""Opt-in finite fault-injection contracts; no CARLA/model/control API."""
import copy
from simulator.m8j13_logic_runtime import canonical

SCENARIOS=('missing_camera','invalid_output','manual_emergency_stop')
REASONS=dict(missing_camera='MissingCamera',invalid_output='ControlOutOfRange',manual_emergency_stop='ManualEmergencyStop')


def scenario(cfg):
    item=cfg.get('fault_injection')
    if item is None:return None
    if item.get('enabled') is not True or item.get('scenario') not in SCENARIOS:raise ValueError('UnknownOrDisabledFaultScenario')
    if item.get('max_injections')!=1:raise ValueError('InjectionBudget')
    return item['scenario']


class CameraSuppression:
    def __init__(self,cfg,identity,actors):
        self.enabled=scenario(cfg)=='missing_camera';self.identity=tuple(identity);self.actors=dict(actors);self.key=None;self.hit=None
    def arm(self,key):
        if not self.enabled or self.key is not None:raise ValueError('DisabledOrDuplicateInjection')
        if tuple(key['identity'])!=self.identity or key['camera']!='CAM_FRONT' or key['actor_id']!=self.actors['CAM_FRONT']:raise ValueError('InjectionIdentityMismatch')
        if not isinstance(key['frame'],int):raise ValueError('InjectionFrame')
        self.key=copy.deepcopy(key)
    def suppress(self,row):
        if self.key is None:return False
        if (row['run'],row['episode'],row['sensor_generation'])!=self.identity:raise ValueError('InjectionIdentityMismatch')
        if row['camera']!=self.key['camera'] or row['frame']!=self.key['frame']:return False
        if row['actor_id']!=self.key['actor_id']:raise ValueError('InjectionIdentityMismatch')
        if self.hit is not None:raise ValueError('DuplicateInjectedCamera')
        self.hit=copy.deepcopy(row);return True


def receive_camera(payload,target,inputs,hook,record):
    """The production drain path: independently copied bytes -> hook -> TargetCapture."""
    row=payload.stamp.record();row['copied_monotonic_s']=payload.copied_monotonic_s
    if (payload.width,payload.height,len(payload.bgra))!=(704,396,704*396*4):raise RuntimeError('CameraPayloadShapeMismatch')
    if hook is not None and hook.suppress(row):
        row['disposition']='TestSuppressed';record(row);return
    outcome=target.camera(row);row['disposition']=outcome;record(row)
    if outcome in ('PendingSnapshot','Selected'):
        inputs.groups[row['frame']].append(row);inputs.payloads[(row['camera'],row['frame'])]=payload.bgra


def collect_input(target,inputs,current,drain,budget,clock,sleep,reserve_s=0.):
    """Reserve diagnostic/return time INSIDE original .5s. No additional ticks."""
    until=min(target.start+.5,budget.operation['deadline'])-reserve_s
    while clock()<until:
        budget.remaining();drain()
        if len(target.groups)==6 and target.target in inputs.imu:break
        sleep(.002)
    budget.remaining()
    state=current()
    try:target.validate(state,clock())
    except ValueError as exc:
        # Preserve every other contract fault; do not disguise mixed metadata.
        if reserve_s>0 and str(exc)=='InputMetadata:MissingCamera':raise ValueError('MissingCamera')
        raise
    return state


def synthetic_reply(task,now,cfg,late=False):
    reply={k:copy.deepcopy(task[k]) for k in ('sequence','identity','frame','source','control')}
    if task.get('injection'):
        if scenario(cfg)!='invalid_output' or task['injection']!='invalid_output':raise ValueError('InjectionScenarioMismatch')
        reply['control']['throttle']=1.1
    reply.update(worker_received_wall_s=task['worker_received_wall_s'],worker_completed_wall_s=now,late=late,model_forward=False)
    return reply


def validate_manual_request(request,state,heartbeat,now):
    if canonical(request.get('identity'))!=canonical(state.get('identity')):raise ValueError('ManualIdentityMismatch')
    if state.get('fault') or state.get('operation') is not None:raise ValueError('ManualInjectionNotQuiescent')
    if state['phase']!='ExpectedComputeWait' or canonical(state['task'])!=canonical(request['task']):raise ValueError('ManualTaskMismatch')
    if now-state['owner_wall_s']>=.3 or now-heartbeat['wall_s']>=.3 or heartbeat['task']!=request['task']['sequence']:raise ValueError('ManualHeartbeatNotLive')
    if now>=request['task']['requested_wall_s']+.5 or now>=request['task']['deadline_wall_s']-1.:raise ValueError('ManualInjectionTooLate')
    if not 1.<=request['forward_speed_mps']<1.7:raise ValueError('ManualSpeedPrecondition')


def reserve_scenario(records,name,run_id,config_sha):
    """Append-only suite journal, first failure prevents all later starts."""
    starts=[r for r in records if r['event']=='reserved']
    if len(starts)>=3 or name!=SCENARIOS[len(starts)]:raise ValueError('SuiteOrderOrStartBudget')
    if starts:
        finishes=[r for r in records if r['event']=='finished']
        if len(finishes)!=len(starts) or not all(r['passed'] for r in finishes):raise ValueError('PriorScenarioNotPassed')
    if any(r['run_id']==run_id for r in starts):raise ValueError('RepeatedRunIdentity')
    return dict(event='reserved',scenario=name,run_id=run_id,config_sha256=config_sha,reserved_ticks=800,starts=1,model_forwards=0,model_controls=0,pid_executions=0,training=0)
