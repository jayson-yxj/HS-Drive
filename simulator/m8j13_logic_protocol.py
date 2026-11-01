"""Design-only synchronous_logic_v1 event model. Standard library only.

No simulator/model imports, network, threads, RPC, sleep or runtime entrypoint.
Command tokens are synthetic dispatch permissions, never VehicleControl calls.
"""
from dataclasses import dataclass
from enum import Enum
import math
import copy

class State(str, Enum):
    INIT='InitializationAndStability'
    CAPTURE='BoundedCapture'
    QUIESCENT='ConfirmedFrozenWorld'
    COMPUTE='ExpectedComputeWait'
    VALIDATE='ValidateOutput'
    EXECUTE='FiniteExecution'
    FAULT='FaultLocked'
    STOPPED='PhysicalStopObserved_SyntheticOnly'
    CLOSED='Closed'

@dataclass(frozen=True)
class Snapshot:
    frame:int
    sim:float
    speed:float  # magnitude in m/s; stop evidence never uses forward speed alone
    confirmed:bool=True
    pose:tuple=(0.,0.,0.,0.)  # synthetic x,y,z,yaw; runtime uses measured rigid pose
    forward_speed:float=0.  # signed feedback is distinct from the speed magnitude

@dataclass(frozen=True)
class Command:
    sequence:int
    epoch:int
    source:str
    control:tuple
    input_frame:int|None
    valid_until_sim:float|None
    valid_until_wall:float

class LogicProtocol:
    """Bounded per-run pure reference. Caller feeds explicit clocks/events.

    Feedback booleans are external assertions in this prototype, not proof of
    physical execution. Fault state never re-arms; create a new explicit run.
    """
    def __init__(self,cfg,snapshot,wall=0.,*,sensor_instances):
        if cfg['protocol_id']!='synchronous_logic_v1' or cfg['execution_enabled'] or cfg['model_actuation_allowed'] or cfg['runtime_accepted']:
            raise ValueError('DesignOnlyConfigurationRequired')
        if set(sensor_instances)!=set(cfg['camera_order']):raise ValueError('SensorRegistryRequired')
        if not self._valid_snapshot(snapshot) or not snapshot.confirmed:raise ValueError('ConfirmedInitialWorldRequired')
        self.sensor_instances=dict(sensor_instances);self.stability_admitted=False;self.generation=None
        self.c=copy.deepcopy(cfg);self.dt=cfg['physics_dt_s'];self.state=State.INIT;self.snap=snapshot;self.wall=wall;self.born=wall
        self.deadline=wall+cfg['wall_deadlines_s']['initialization'];self.heartbeat=wall
        self.permits={};self.fault=None;self.epoch=0;self.seq=0;self.active=None;self.sent=None;self.safety_sent=None;self.safety_rpc_start=None;self.delivered=None;self.rpc_start=None
        self.tick_pending=None;self.tick_start=None;self.target=None;self.target_start=None;self.target_sim=None;self.target_snapshot=None;self.cameras={};self.imu=False
        self.input=None;self.plan=None;self.result_seen=set();self.execution_ticks=0;self.physical_ticks=0;self.fault_ticks=0;self.stop_start=None;self.stop_hold=0.
        self.delivery_status='NotAttempted';self.events=[];self.pid_calls=0;self.interventions=0;self.runtime_calls=0
    @staticmethod
    def _valid_snapshot(snapshot):
        return isinstance(snapshot.frame,int) and snapshot.speed>=0 and len(snapshot.pose)==4 and all(math.isfinite(x) for x in (snapshot.sim,snapshot.speed,snapshot.forward_speed,*snapshot.pose))
    def _event(self,kind,**data):self.events.append(dict(event=kind,wall=self.wall,sim=self.snap.sim,state=self.state.value,**data))
    def _transition(self,state,deadline):self.state=state;self.deadline=deadline;self._event('state')
    def fault_stop(self,reason):
        if self.fault is None:
            self.fault=reason;self.epoch+=1;self.active=None;self.plan=None;self.interventions+=1
            self.delivery_status='ResponseUnverified' if self.sent is not None or self.tick_pending is not None else 'NotAttempted'
            self._transition(State.FAULT,self.wall+self.c['wall_deadlines_s']['fault_stop_observation']);self._event('fault',reason=reason)
        return False
    def advance(self,wall,heartbeat=False):
        if not math.isfinite(wall) or wall<self.wall:return self.fault_stop('InvalidWallClock')
        self.wall=wall
        # Ordinary heartbeat cannot rescue a deadline already missed.
        if self.state not in (State.CLOSED,State.STOPPED,State.FAULT):
            if wall-self.heartbeat>self.c['wall_deadlines_s']['owner_heartbeat']+1e-9:return self.fault_stop('OwnerUnresponsive')
            if wall>self.born+self.c['wall_deadlines_s']['whole_run']:return self.fault_stop('RunDeadline')
            if wall>self.deadline:return self.fault_stop('PhaseDeadline:'+self.state.value)
        if heartbeat:self.heartbeat=wall
        if self.state==State.FAULT and wall>self.deadline:
            self._event('stop_observation_exhausted',response=self.delivery_status);self.state=State.CLOSED;return False
        if self.safety_sent is not None and wall-self.safety_rpc_start>self.c['wall_deadlines_s']['command_delivery']:
            self.fault_stop('SafetyCommandRPCDeadline');self.delivery_status='ResponseUnverified';return False
        if self.tick_pending is not None and wall-self.tick_start>self.c['wall_deadlines_s']['tick_rpc']:
            self.fault_stop('TickRPCDeadline');self.delivery_status='ResponseUnverified';return False
        if self.sent is not None and wall-self.rpc_start>self.c['wall_deadlines_s']['command_delivery']:
            self.fault_stop('CommandRPCDeadline');self.delivery_status='ResponseUnverified';return False
        return self.fault is None
    def stable(self,samples,wall):
        if not self.advance(wall,True) or self.state!=State.INIT:return False
        if self.stability_admitted:return self.fault_stop('RepeatedStabilityAdmission')
        if len(samples)!=self.c['stable_ticks']:return self.fault_stop('InsufficientStabilityEvidence')
        previous=self.snap
        for sample in samples:
            if not self._valid_snapshot(sample) or not sample.confirmed or sample.frame!=previous.frame+1 or abs(sample.sim-previous.sim-self.dt)>1e-6 or abs(sample.speed)>self.c['stop_speed_mps']:return self.fault_stop('InvalidStabilityEvidence')
            previous=sample
        if self.physical_ticks+len(samples)>self.c['max_initialization_ticks']:return self.fault_stop('InitializationTickCap')
        self.stability_admitted=True;self.snap=previous;self.physical_ticks+=len(samples);self._event('initial_stability_synthetic');return True
    def begin_capture(self,wall,generation):
        if not self.advance(wall,True):return False
        if self.state==State.INIT:
            if not self.stability_admitted:return self.fault_stop('NotStable')
            target=self.snap.frame+1
        elif self.state==State.EXECUTE:
            if self.execution_ticks!=self.c['max_execution_ticks_before_next_target']:return self.fault_stop('WrongCaptureCadence')
            target=self.snap.frame
            if target!=self.input['frame']+self.c['observation_interval_ticks']:return self.fault_stop('SkippedTarget')
        else:return self.fault_stop('InvalidCaptureState')
        if not isinstance(generation,tuple) or len(generation)!=3:return self.fault_stop('RunEpisodeGenerationRequired')
        if self.generation is not None and self.generation!=generation:return self.fault_stop('ChangedGeneration')
        self.target=target;self.generation=generation;self.target_sim=None;self.target_snapshot=None;self.cameras={};self.imu=False
        # Next target was just ticked under previous plan; anchor to tick initiation.
        self.target_start=self.last_tick_start if target==self.snap.frame else wall
        if target==self.snap.frame:self.target_sim=self.snap.sim;self.target_snapshot=self.snap
        self._transition(State.CAPTURE,self.target_start+self.c['wall_deadlines_s']['capture']);return True
    def sensor(self,camera,frame,sim,generation,actor_id,wall):
        if not self.advance(wall):return False
        if self.state!=State.CAPTURE:return self.fault_stop('LateSensor')
        if camera not in self.c['camera_order'] or actor_id!=self.sensor_instances.get(camera):return self.fault_stop('WrongSensorInstance')
        if camera in self.cameras:return self.fault_stop('DuplicateCamera')
        if not math.isfinite(sim) or frame!=self.target or generation!=self.generation or self.target_sim is None or abs(sim-self.target_sim)>self.c['timestamp_tolerance_s']:return self.fault_stop('CrossFrameOrGeneration')
        self.cameras[camera]=(frame,sim,generation,actor_id);return True
    def ego_metadata(self,frame,sim,imu_frame,imu_sim,generation,navigation_valid,wall):
        if not self.advance(wall) or self.state!=State.CAPTURE:return False
        if not all(math.isfinite(x) for x in (sim,imu_sim)) or frame!=self.target or imu_frame!=frame or generation!=self.generation or not navigation_valid or self.target_sim is None or abs(sim-self.target_sim)>1e-5 or abs(imu_sim-sim)>1e-5:return self.fault_stop('EgoIMUNavigationMismatch')
        self.imu=True;return True
    def confirm_quiescent(self,snapshot,wall,no_pending_tick):
        if not self.advance(wall,True):return False
        if self.state!=State.CAPTURE or self.tick_pending is not None or not no_pending_tick or not snapshot.confirmed or snapshot!=self.snap:return self.fault_stop('UnconfirmedWorld')
        if self.snap.frame!=self.target+self.c['capture_flush_ticks']:return self.fault_stop('CaptureFlushContract')
        if set(self.cameras)!=set(self.c['camera_order']) or not self.imu:return self.fault_stop('MissingInput')
        self.input=dict(frame=self.target,sim=self.target_sim,wall=self.target_start,generation=self.generation)
        self._transition(State.QUIESCENT,self.wall+self.c['wall_deadlines_s']['quiesce_and_hold_delivery']);return True
    def begin_compute(self,wall):
        if not self.advance(wall,True):return False
        if self.state!=State.QUIESCENT or self.tick_pending is not None or self.sent is not None or self.safety_sent is not None or self.delivered is None or self.delivered.source!='PlannedHoldBrake':return self.fault_stop('UnconfirmedComputeHold')
        self.frozen=self.snap;self._transition(State.COMPUTE,min(self.wall+self.c['wall_deadlines_s']['compute'],self.input['wall']+self.c['wall_deadlines_s']['observation_to_last_command_effect']));return True
    def receive_result(self,result_id,input_frame,snapshot,wall,trajectory_valid=True,*,generation):
        if not self.advance(wall,True):return False
        if self.state!=State.COMPUTE or result_id in self.result_seen:return self.fault_stop('DuplicateOrUnexpectedResult')
        if not self._valid_snapshot(snapshot) or snapshot!=self.frozen or not snapshot.confirmed or self.tick_pending is not None:return self.fault_stop('UnexpectedWorldAdvanceOrUnconfirmed')
        if generation!=self.input['generation'] or input_frame!=self.input['frame'] or snapshot.sim-self.input['sim']>self.c['max_observation_age_sim_s']+1e-6:return self.fault_stop('ExpiredOrWrongInput')
        if not trajectory_valid:return self.fault_stop('InvalidTrajectory')
        self.result_seen.add(result_id);self._transition(State.VALIDATE,self.wall+self.c['wall_deadlines_s']['validation']);return True
    def accept_pid(self,raw,wall,future_speed_max=0.):
        if not self.advance(wall,True):return False
        if self.state!=State.VALIDATE:return self.fault_stop('UnexpectedPIDUpdate')
        if not self._valid_control(raw):return self.fault_stop('NonfiniteOrInvalidControl')
        self.pid_calls+=1;self.plan=dict(self.input,raw=tuple(raw));self.execution_ticks=0
        self._event('one_pid_update',saturated=raw[1]==1.,future_speed_above_cap=future_speed_max>self.c['max_speed_mps'])
        self._transition(State.EXECUTE,self.wall+self.c['wall_deadlines_s']['execute_to_next_target']);return True
    @staticmethod
    def _valid_control(raw):
        try:
            return len(raw)==3 and all(math.isfinite(x) for x in raw) and -1<=raw[0]<=1 and 0<=raw[1]<=1 and 0<=raw[2]<=1 and not(raw[1]>0 and raw[2]>0)
        except (TypeError,ValueError):return False
    def command(self,wall):
        self.advance(wall,True)
        if self.state==State.CLOSED:return None
        if self.fault:source='FaultBrake';raw=(0.,0.,1.);p=None
        elif self.state in (State.QUIESCENT,State.COMPUTE,State.VALIDATE):source='PlannedHoldBrake';raw=(0.,0.,1.);p=None
        elif self.state==State.INIT or self.plan is None:source='InitialProtectionBrake';raw=(0.,0.,1.);p=None
        else:source='ModelHeldCommand';raw=self.plan['raw'];p=self.plan
        self.seq+=1
        token=Command(self.seq,self.epoch,source,raw,None if p is None else p['frame'],None if p is None else p['sim']+self.c['max_observation_age_sim_s'],
            self.wall+self.c['wall_deadlines_s']['command_delivery'] if p is None else p['wall']+self.c['wall_deadlines_s']['observation_to_last_command_effect'])
        self.permits={token.sequence:token}
        return token
    def dispatch(self,command,wall):
        self.advance(wall,True)
        if command is None or self.state==State.CLOSED:return False
        if command.epoch!=self.epoch or (self.fault and command.source!='FaultBrake'):
            self._event('stale_command_rejected');return False
        if self.permits.get(command.sequence)!=command:return self.fault_stop('UnknownOrModifiedCommand')
        if command.source=='ModelHeldCommand' and (self.state not in (State.EXECUTE,State.CAPTURE) or self.plan is None or command.input_frame!=self.plan['frame']):return self.fault_stop('SupersededCommand')
        safety=command.source=='FaultBrake'
        if (self.safety_sent if safety else self.sent) is not None:return self.fault_stop('CommandAlreadyInFlight')
        if command.sequence<=getattr(self,'last_dispatched',0):return self.fault_stop('ReplayedCommand')
        if wall>command.valid_until_wall:return self.fault_stop('CommandWallExpired')
        if safety:self.safety_sent=command;self.safety_rpc_start=wall
        else:self.sent=command;self.rpc_start=wall
        self.last_dispatched=command.sequence;self.permits.pop(command.sequence);self.delivery_status='InFlight';self._event('synthetic_dispatch',source=command.source);return True
    def command_return(self,sequence,wall,delivered=True):
        self.advance(wall,True)
        if self.safety_sent is not None and sequence==self.safety_sent.sequence:
            command=self.safety_sent;self.safety_sent=None
        elif self.sent is not None and sequence==self.sent.sequence:
            command=self.sent;self.sent=None
        else:return self.fault_stop('UnexpectedCommandReturn')
        if not delivered:
            self.fault_stop('CommandNotDelivered');self.delivery_status='CommandNotDelivered';return False
        if command.epoch!=self.epoch or (self.fault and command.source!='FaultBrake'):
            self.active=None;self.delivery_status='ResponseUnverified';self._event('late_RPC_requires_fresh_brake');return False
        self.delivered=command;self.active=command;self.delivery_status='ResponseUnverified';return True
    def begin_tick(self,wall):
        self.advance(wall,True)
        if self.state in (State.CLOSED,State.STOPPED):return False
        if self.tick_pending is not None:return self.fault_stop('ConcurrentTickOrBacklog')
        if self.state not in (State.INIT,State.CAPTURE,State.EXECUTE,State.FAULT):return self.fault_stop('TickDuringExpectedWait')
        if self.sent is not None or self.safety_sent is not None or self.active is None:return self.fault_stop('ControlNotDelivered')
        if self.fault and self.active.source!='FaultBrake':return False
        if not self.fault and self.plan is not None and self.active.source!='ModelHeldCommand':return self.fault_stop('NormalHandoffUnconfirmed')
        if not self.fault and self.plan is not None and self.active.input_frame!=self.plan['frame']:return self.fault_stop('WrongPlanCommand')
        if self.state==State.INIT and self.physical_ticks>=self.c['max_initialization_ticks']:return self.fault_stop('InitializationTickCap')
        next_frame=self.snap.frame+1;next_sim=self.snap.sim+self.dt
        if self.state==State.CAPTURE and next_frame>self.target+self.c['capture_flush_ticks']:return self.fault_stop('CaptureTickCap')
        if self.state==State.EXECUTE and self.execution_ticks>=self.c['max_execution_ticks_before_next_target']:return self.fault_stop('ExecutionTickCap')
        if self.state==State.FAULT and self.fault_ticks>=self.c['fault_stop_max_ticks']:self.state=State.CLOSED;return False
        if self.active.source=='ModelHeldCommand':
            if self.wall+self.c['wall_deadlines_s']['tick_rpc']>self.active.valid_until_wall or next_sim>self.active.valid_until_sim+1e-6:return self.fault_stop('OriginalCommandValidityExhausted')
        if self.state==State.CAPTURE and next_frame==self.target:self.target_start=wall;self.deadline=wall+self.c['wall_deadlines_s']['capture']
        if self.state==State.FAULT:self.fault_ticks+=1
        self.tick_pending=next_frame;self.tick_start=wall;self.last_tick_start=wall;return True
    def tick_return(self,snapshot,wall,feedback_matches=True):
        self.advance(wall,True)
        if not self._valid_snapshot(snapshot) or self.tick_pending is None or snapshot.frame!=self.tick_pending or not snapshot.confirmed or abs(snapshot.sim-self.snap.sim-self.dt)>1e-6:return self.fault_stop('UnconfirmedOrUnexpectedTick')
        self.tick_pending=None;self.snap=snapshot;self.physical_ticks+=1
        if not feedback_matches:
            self.stop_start=None;self.stop_hold=0.;self.delivery_status='ResponseUnverified'
            return self.fault_stop('ControlFeedbackMismatch')
        if self.active is None or (self.fault and self.active.source!='FaultBrake'):
            self.delivery_status='ResponseUnverified';return False
        if self.active.source=='ModelHeldCommand' and wall>self.active.valid_until_wall:return self.fault_stop('CommandEffectWallExpired')
        self.delivery_status='ControlFeedbackObserved'
        if self.state==State.CAPTURE and snapshot.frame==self.target:self.target_sim=snapshot.sim;self.target_snapshot=snapshot
        if self.state==State.EXECUTE:self.execution_ticks+=1
        if not self.fault and snapshot.forward_speed<-.2:return self.fault_stop('ReverseMotion')
        if not self.fault and snapshot.speed>=self.c['speed_guard_trigger_mps']:return self.fault_stop('IndependentSpeedGuard')
        if self.state==State.FAULT:
            if abs(snapshot.speed)<=self.c['stop_speed_mps']:
                if self.stop_start is None:self.stop_start=snapshot.sim
                self.stop_hold=snapshot.sim-self.stop_start
                if self.stop_hold>=self.c['stop_hold_sim_s']-1e-6:self.state=State.STOPPED;self.delivery_status='PhysicalStopObserved_SyntheticOnly'
            else:self.stop_start=None;self.stop_hold=0.
        return True
    def connection_lost(self,wall):
        self.advance(wall);self.fault_stop('ConnectionLost');self.delivery_status='ResponseUnverified' if self.sent or self.safety_sent or self.tick_pending else 'CommandNotDelivered'
    def manual_stop(self,wall):self.advance(wall);return self.fault_stop('ManualEmergencyStop')
    def finish(self,wall):
        if not self.advance(wall,True):return False
        if self.tick_pending or self.sent or self.safety_sent:return self.fault_stop('FinishWithInFlightOperation')
        self.fault_stop('NormalTerminationBrakeRequired');return True
