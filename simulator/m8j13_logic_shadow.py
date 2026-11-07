"""Explicit logical shadow contracts. No model/CARLA/actuation imports."""
import copy
import math
from simulator.m8j13_logic_runtime import canonical,frozen_equal,STOP

PROTOCOL='synchronous_logic_v1_parked_shadow'
CHECKS={'input','finite','shape','trajectory','pid_raw','agent_control','parameters','memory'}
ENDPOINTS=('worker_received','read_done','preprocess_start','preprocess_done','transfer_done','forward_start','forward_done','hs_done','transport_done','pid_done','checks_done','publish_start')
FIELDS={'protocol','identity','sequence','frame','input_sim_s','input_wall_s','task_deadline','validation_start','validation_deadline','checks','times','model_forward'}


def require_parked_execution(cfg,command,tick=False):
    if cfg.get('shadow_protocol')!=PROTOCOL:return
    if command['source'] not in ('InitialProtectionBrake','PlannedHoldBrake','FaultBrake') or canonical(command['control'])!=canonical(STOP):raise ValueError('ShadowExecutionMustBeIndependentBrake')
    if tick and command['source']=='PlannedHoldBrake':raise ValueError('HoldCannotAdvancePhysics')


def task_valid(task,now,current=None,pending=False):
    if task['protocol']!=PROTOCOL:raise ValueError('WrongShadowProtocol')
    if pending:raise ValueError('ShadowOperationInFlight')
    if now>min(task['deadline_wall_s'],task['input_wall_s']+4.,task['run_deadline']):raise ValueError('ShadowTaskDeadline')
    if current is not None:
        if not frozen_equal(task['frozen'],current):raise ValueError('ShadowWorldChanged')
        age=current['sim_s']-task['input_sim_s']
        if not 0<=age<=.3+1e-6:raise ValueError('ShadowSimulationAge')


def require_notification(reply):
    if set(reply)!=FIELDS or set(reply['checks'])!=CHECKS or set(reply['times'])!=set(ENDPOINTS):raise ValueError('ShadowCompletionSchema')
    if reply['protocol']!=PROTOCOL or reply['model_forward'] is not True or not all(reply['checks'][k] is True for k in CHECKS):raise ValueError('ShadowNecessaryChecksFailed')
    values=list(reply['times'].values())+[reply[k] for k in ('input_sim_s','input_wall_s','task_deadline','validation_start','validation_deadline')]
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values):raise ValueError('ShadowCompletionClock')
    ordered=[reply['times'][k] for k in ENDPOINTS]
    if ordered!=sorted(ordered):raise ValueError('ShadowEndpointOrder')
    if reply['validation_start']!=reply['times']['forward_done']:raise ValueError('ValidationStartChanged')


class ShadowChannel:
    def __init__(self,identity):self.identity=list(identity);self.sequence=0;self.task=None;self.consumed=set();self.locked=False;self.previous=None
    def request(self,observation,frozen,now,run_deadline):
        if self.locked or self.sequence>=24 or (self.task and self.sequence not in self.consumed):raise ValueError('ShadowLockedPendingOrBudget')
        if self.previous and (observation['frame']!=self.previous['frame']+4 or abs(observation['sim_s']-self.previous['sim_s']-.2)>1e-5):raise ValueError('ShadowInputCadence')
        self.sequence+=1
        self.task=dict(protocol=PROTOCOL,identity=list(self.identity),sequence=self.sequence,frame=observation['frame'],input_sim_s=observation['sim_s'],input_wall_s=observation['input_wall_s'],requested_wall_s=now,deadline_wall_s=min(now+2.,observation['input_wall_s']+4.,run_deadline),run_deadline=run_deadline,frozen=copy.deepcopy(frozen))
        task_valid(self.task,now,frozen);self.previous=copy.deepcopy(observation)
        return copy.deepcopy(self.task)
    def consume(self,reply,current,now,pending=False):
        if self.locked:raise ValueError('LateShadowLocked')
        require_notification(reply);t=self.task
        if t is None or reply['sequence']!=t['sequence'] or reply['sequence'] in self.consumed:raise ValueError('ShadowRepeatedOrWrongSequence')
        for key in ('protocol','identity','sequence','frame','input_sim_s','input_wall_s'):
            if canonical(reply[key])!=canonical(t[key]):raise ValueError('ShadowIdentityMismatch')
        if reply['task_deadline']!=t['deadline_wall_s']:raise ValueError('ShadowDeadlineResigned')
        expected=min(reply['validation_start']+.1,t['deadline_wall_s'],t['input_wall_s']+4.,t['run_deadline'])
        if reply['validation_deadline']!=expected or now>expected:raise ValueError('ShadowValidationDeadline')
        if reply['times']['publish_start']>now or reply['times']['worker_received']<t['requested_wall_s']:raise ValueError('ShadowEndpointClock')
        task_valid(t,now,current,pending)
        return dict(sequence=t['sequence'],frame=t['frame'],sim_age_s=current['sim_s']-t['input_sim_s'])
    def finalize_return(self,reply,returned):
        # Caller samples actual consume() return, then commits single consumption.
        if self.locked or reply['sequence'] in self.consumed:raise ValueError('LateOrDuplicateReturn')
        task_valid(self.task,returned)
        if returned>reply['validation_deadline']:raise ValueError('ShadowValidationDeadlineAtReturn')
        self.consumed.add(reply['sequence'])


def memory_valid(before,after,previous,timestamp):
    old=before['history_timestamp'];new=after['history_timestamp']
    if previous is None:
        if old is not None or before['cached']:raise ValueError('CrossRunModelMemory')
    elif old is None or any(abs(t-previous)>1e-5 for t in old):raise ValueError('PreviousModelMemoryMismatch')
    if old is not None and any(t>=timestamp for t in old):raise ValueError('FutureModelMemory')
    if new is None or any(abs(t-timestamp)>1e-5 for t in new):raise ValueError('CurrentModelMemoryMismatch')
