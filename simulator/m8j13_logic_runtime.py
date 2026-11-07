"""Small no-model runtime guards. No simulator/model/PID imports or actuation."""
import copy
import math
from collections import Counter
from simulator.m8j13_sensor_audit import check_group
from simulator.m8j13_safety import sanitize_control

SOURCES = {'IndependentTestControl', 'InitialProtectionBrake', 'PlannedHoldBrake', 'FaultBrake'}
STOP = dict(steer=0., throttle=0., brake=1., hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)


def canonical(value):
    if isinstance(value, (tuple, list)): return tuple(canonical(x) for x in value)
    if isinstance(value, dict): return tuple(sorted((k, canonical(v)) for k, v in value.items()))
    if isinstance(value, float) and not math.isfinite(value): raise ValueError('NonfiniteState')
    return value


def frozen_equal(first, current):
    return all(canonical(first[k]) == canonical(current[k]) for k in ('frame','sim_s','pose','velocity'))


def supervision_reason(state, now, manual=False):
    if manual: return 'ManualEmergencyStop'
    if not state.get('armed'): return None
    if now - state['owner_wall_s'] > .3: return 'OwnerUnresponsive'
    if state.get('fault'): return state['fault']
    if now > state['run_deadline']: return 'RunDeadline'
    if now > state['phase_deadline']: return 'TaskDeadline' if state['phase']=='ExpectedComputeWait' else 'PhaseDeadline:'+state['phase']
    op=state.get('operation')
    if op and now>op['deadline']: return 'TickRPCDeadline' if op['kind']=='tick' else 'CommandRPCDeadline'
    return None


def reserve_start(ledger, cap=800):
    ledger=copy.deepcopy(ledger or {'runs':[]})
    if ledger['runs']: raise ValueError('StartBudgetExhausted')
    ledger['runs'].append(dict(reserved_ticks=cap,starts=1,model_forwards=0,model_controls=0,training=0))
    return ledger


class TargetCapture:
    """Registered before tick, buffer before snapshot, immutable selected identity."""
    def __init__(self, identity, actors):
        self.identity=tuple(identity);self.actors=dict(actors);self.target=None;self.groups={};self.snapshot=None
        self.selected=set();self.stats=Counter();self.start=None;self.accepted=False;self.closed=False
    def register(self, frame):
        if self.closed or frame in self.selected: raise ValueError('RepeatedOrClosedTarget')
        self.target=frame;self.selected.add(frame);self.groups={};self.snapshot=None;self.start=None;self.accepted=False
    def start_tick(self, frame, wall):
        if frame==self.target:
            if self.start is not None: raise ValueError('RepeatedTargetTick')
            self.start=wall
    def camera(self, row):
        if self.closed: self.stats['AfterClose']+=1;return 'AfterClose'
        if (row['run'],row['episode'],row['sensor_generation'])!=self.identity: raise ValueError('OldGeneration')
        if row['frame']!=self.target:
            if row['frame'] in self.selected: raise ValueError('LateSelectedObservation')
            self.stats['NotSelected']+=1;return 'NotSelected'
        if row['camera'] not in self.actors or self.actors[row['camera']]!=row['actor_id']: raise ValueError('WrongSensorInstance')
        if row['camera'] in self.groups: raise ValueError('DuplicateCamera')
        self.groups[row['camera']]=copy.deepcopy(row);return 'PendingSnapshot' if self.snapshot is None else 'Selected'
    def confirm(self, row):
        if row['frame']==self.target:
            if self.snapshot is not None: raise ValueError('DuplicateSnapshot')
            self.snapshot=copy.deepcopy(row)
    def validate(self, current, now, pending=False):
        if pending or self.snapshot is None: raise ValueError('SnapshotUnconfirmed')
        if current['frame']!=self.target+2: raise ValueError('CaptureFlushContract')
        if self.start is None or now-self.start>.5: raise ValueError('CaptureDeadline')
        check=check_group(list(self.groups.values()),dict(frame=self.target,elapsed_seconds=self.snapshot['sim_s']),self.identity,self.actors)
        if not check['valid']: raise ValueError('InputMetadata:'+','.join(check['errors']))
        if abs(current['sim_s']-self.snapshot['sim_s']-.1)>1e-5: raise ValueError('CaptureSimulationAge')
        self.accepted=True;return check
    def close(self): self.closed=True;self.groups.clear()


class TaskChannel:
    def __init__(self, identity):self.identity=tuple(identity);self.sequence=0;self.task=None;self.consumed=set();self.locked=False
    def request(self, capture, frozen, now, mode, control):
        if self.locked or (self.task and self.task['sequence'] not in self.consumed):raise ValueError('TaskAlreadyPendingOrLocked')
        self.sequence+=1
        self.task=dict(sequence=self.sequence,identity=list(self.identity),frame=capture['frame'],input_sim_s=capture['sim_s'],input_wall_s=capture['input_wall_s'],
                       requested_wall_s=now,deadline_wall_s=now+2.,mode=mode,source='IndependentTestControl',control=dict(control),frozen=copy.deepcopy(frozen))
        return copy.deepcopy(self.task)
    def consume(self, result, current, now, pending=False):
        if self.locked:raise ValueError('LateResultLocked')
        task=self.task
        if task is None or result['sequence']!=task['sequence'] or result['sequence'] in self.consumed:raise ValueError('UnexpectedOrRepeatedResult')
        if tuple(result['identity'])!=self.identity or result['frame']!=task['frame'] or result['source']!='IndependentTestControl':raise ValueError('WrongTaskIdentityOrSource')
        if now>task['deadline_wall_s']:raise ValueError('TaskDeadline')
        if pending or not frozen_equal(task['frozen'],current):raise ValueError('UnconfirmedOrChangedFrozenWorld')
        if current['sim_s']-task['input_sim_s']>.3+1e-6 or now>task['input_wall_s']+4.:raise ValueError('OriginalAgeExpired')
        _,error=sanitize_control(result.get('control'))
        if error:raise ValueError(error)
        if canonical(result['control'])!=canonical(task['control']):raise ValueError('SyntheticControlChanged')
        control=result['control']
        if not all(math.isfinite(control[k]) for k in ('steer','throttle','brake')):raise ValueError('NonfiniteControl')
        self.consumed.add(task['sequence']);return copy.deepcopy(control)


class CommandGuard:
    def __init__(self):self.epoch=0;self.sequence=0;self.locked=False;self.pending=None;self.active=None;self.uncertain=False
    def prepare(self, source, control, wall, observation=None):
        if source not in SOURCES or (self.locked and source!='FaultBrake'):raise ValueError('ForbiddenOrLockedControlSource')
        if self.pending is not None:raise ValueError('CommandInFlight')
        if source!='IndependentTestControl' and canonical(control)!=canonical(STOP):raise ValueError('ProtectionMustBrake')
        self.sequence+=1
        cmd=dict(sequence=self.sequence,epoch=self.epoch,source=source,control=copy.deepcopy(control),request_wall_s=wall,observation=copy.deepcopy(observation))
        self.pending=cmd;return copy.deepcopy(cmd)
    def confirm(self, command, success):
        if self.pending is None or canonical(command)!=canonical(self.pending):raise ValueError('UnknownCommandConfirmation')
        self.pending=None
        if not success:self.uncertain=True;raise ValueError('HandoffUncertain')
        if command['epoch']!=self.epoch:self.active=None;raise ValueError('LateNormalConfirmation')
        self.active=command;return 'ServerProcessed_PhysicsUnverified'
    def latch(self):self.locked=True;self.epoch+=1;self.active=None
    def before_tick(self, next_sim, wall):
        if self.pending or self.uncertain or self.active is None:raise ValueError('HandoffUncertain')
        cmd=self.active
        if self.locked and cmd['source']!='FaultBrake':raise ValueError('LockedNormalCommand')
        if cmd['source']=='PlannedHoldBrake':raise ValueError('HoldCannotAdvancePhysics')
        obs=cmd['observation']
        if obs and (next_sim>obs['sim_s']+.3+1e-6 or wall+.5>obs['input_wall_s']+4.):raise ValueError('OriginalCommandExpired')
        return cmd


class OperationDeadline:
    """One immutable absolute monotonic deadline, including queue and processing."""
    def __init__(self, operation, clock):
        self.operation=dict(operation);self.clock=clock
    def remaining(self):
        left=self.operation['deadline']-self.clock()
        if not math.isfinite(left) or left<=0:raise TimeoutError('OperationDeadline:'+self.operation['kind'])
        return left
    def call(self, function, *args, client=None, timeout_argument=False, **kwargs):
        left=self.remaining()
        if client is not None:client.set_timeout(left)
        if timeout_argument:kwargs['seconds']=self.remaining()
        self.remaining()
        value=function(*args,**kwargs)
        self.remaining()
        return value


def operation_stamp(sequence, kind, now, limit, phase_deadline, run_deadline, extra_deadline=None):
    deadlines=[now+limit,phase_deadline,run_deadline]
    if extra_deadline is not None:deadlines.append(extra_deadline)
    deadline=min(deadlines)
    if deadline<=now:raise TimeoutError('OperationDeadlineBeforeSubmission:'+kind)
    return dict(sequence=sequence,kind=kind,started=now,deadline=deadline,operation_limit_s=limit,
                phase_deadline=phase_deadline,run_deadline=run_deadline)


def initialization_due(last_actual_rpc_start, minimum_interval=.05):
    return None if last_actual_rpc_start is None else last_actual_rpc_start+minimum_interval
