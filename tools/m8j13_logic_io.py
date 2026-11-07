"""Deadline-aware setup, tick and cleanup adapters for the explicit test only.

Legacy Owner and default runtime paths are unchanged. All potentially blocking
CARLA calls here consume the same operation deadline supplied by the owner.
"""
import json
import time
from pathlib import Path
import numpy as np
from tools.m8j13_compatibility import Owner,nav_geometry
from tools.m8j13_common import config,write,append,atomic
from tools.m8j13_sync_diagnostic import settings_record
from simulator.m8j13_carla import measurement
from simulator.m8j13_execution import control_record


class DeadlineOwner(Owner):
    def __init__(self,root,cfg,budget):
        import carla
        self.root=root;self.cfg=cfg;self.c=config();self.actors=[];self.sensors=[];self.clients=[];self.tick_count=0;self.last=None
        self.server_advance_unknown=False;self.budget=budget;self.client=carla.Client('127.0.0.1',self.c['port']);self.world=None;self.original=None;self.weather=None
    def rpc(self,function,*args,**kwargs):return self.budget.call(function,*args,client=self.client,**kwargs)
    def setup(self):
        import carla,sys
        self.world=self.rpc(self.client.get_world)
        versions=[self.rpc(self.client.get_client_version),self.rpc(self.client.get_server_version)]
        if versions!=[self.cfg['version']]*2:raise RuntimeError('ClientServerVersionMismatch')
        self.original=self.rpc(self.world.get_settings);self.weather=self.rpc(self.world.get_weather)
        if self.original.synchronous_mode:raise RuntimeError('WorldAlreadySynchronous')
        if self.rpc(self.world.get_map).name.split('/')[-1]!=self.c['map']:
            self.world=self.rpc(self.client.load_world,self.c['map'])
            self.original=self.rpc(self.world.get_settings);self.weather=self.rpc(self.world.get_weather)
        self.clients.append(dict(role='deadline_io',python=sys.executable,carla_path=carla.libcarla.__file__,client_version=versions[0],server_version=versions[1],port=self.c['port'],episode=self.world.id))
        append(self.root/'clients.jsonl',self.clients[-1])
        self.event('initial_world',settings=settings_record(self.world),episode=self.world.id)
        s=self.rpc(self.world.get_settings);s.synchronous_mode=True;s.fixed_delta_seconds=.05;s.no_rendering_mode=False;s.substepping=True;s.max_substeps=10;s.max_substep_delta_time=.01
        before=self.rpc(self.world.get_snapshot).frame
        self.server_advance_unknown=True
        self.rpc(self.world.apply_settings,s,timeout_argument=True)
        after=self.rpc(self.world.get_snapshot).frame;self.server_advance_unknown=False
        actual=settings_record(self.world);self.budget.remaining()
        if actual!=dict(synchronous_mode=True,fixed_delta_seconds=.05,no_rendering_mode=False,substepping=True,max_substep_delta_time=.01,max_substeps=10):raise RuntimeError('SettingsMismatch')
        self.event('settings_applied',settings=actual,observed_frame_delta=after-before,internal_tick_charge=self.cfg['tick_caps']['settings_startup'])
        self.rpc(self.world.set_weather,carla.WeatherParameters.ClearNoon);self.last=after
        route=json.loads(Path(self.c['scenes'][0]['route_file']).read_text());i=self.c['scenes'][0]['start_index'];xyz=route['route_world_carla'][i]
        self.spawn_tf=carla.Transform(carla.Location(x=xyz[0],y=xyz[1],z=xyz[2]+.5),carla.Rotation(yaw=float(-np.degrees(route['route_world_hs'][i][2]))))
        self.target=self.spawn_tf.transform(carla.Location(x=self.cfg['fixed_target_ahead_m']))
        others=self.rpc(self.world.get_actors)
        if list(others.filter('vehicle.*'))+list(others.filter('walker.*')):raise RuntimeError('TestAreaHasOtherTrafficParticipants')
        write(self.root/'geometry.json',nav_geometry(self.world,self.spawn_tf,self.target));self.budget.remaining()
    def spawn(self,bp,tf,parent=None):
        import carla
        args={} if parent is None else dict(attach_to=parent,attachment_type=carla.AttachmentType.Rigid)
        a=self.rpc(self.world.spawn_actor,bp,tf,**args);self.actors.append(a)
        if a.type_id.startswith('sensor.'):self.sensors.append(a)
        self.event('spawn',actor=a.id,type=a.type_id);return a
    def ego(self):
        return self.spawn(self.rpc(self.world.get_blueprint_library).find(self.c['vehicle']),self.spawn_tf)


def measured_tick(owner,ego,budget,reply,before):
    """Called after queue/drain; timeout passed to RPC is remaining, never reset."""
    budget.remaining();reply['rpc_start_wall_s']=time.monotonic()
    try:frame=budget.call(owner.world.tick,client=owner.client,timeout_argument=True)
    finally:reply['rpc_return_wall_s']=time.monotonic()
    snap=budget.call(owner.world.get_snapshot,client=owner.client);reply['snapshot_available_wall_s']=time.monotonic()
    if frame!=snap.frame or frame!=before+1:raise RuntimeError('CompetingTickOwner')
    row=measurement(snap,ego);budget.remaining()
    row['full_control']=control_record(budget.call(ego.get_control,client=owner.client));reply['feedback_complete_wall_s']=time.monotonic()
    return snap,row


def cleanup_owner(owner,budget,root,advance_unknown=False):
    """No new tick. Unknown prior server advance prevents settings restoration."""
    advance_unknown=advance_unknown or getattr(owner,'server_advance_unknown',False)
    report=dict(sensor_stops=[],actor_destruction=[],settings_restored=False,remaining=None,
                advance_state_unknown=advance_unknown,traffic_manager='NotCreated',errors=[])
    def save():atomic(root/'cleanup_steps.json',report)
    save()
    if owner is None or owner.world is None:report['scope']='NoWorldOwned';save();return report
    owner.budget=budget
    try:
        for sensor in owner.sensors:
            row=dict(actor_id=sensor.id,start_wall_s=time.monotonic());report['sensor_stops'].append(row);save()
            try:
                row['before_listening']=budget.call(sensor.is_listening)
                budget.call(sensor.stop,client=owner.client)
                row['after_listening']=budget.call(sensor.is_listening);row['passed']=not row['after_listening']
            except Exception as exc:row.update(passed=False,error=repr(exc));report['errors'].append('SensorStop')
            row['end_wall_s']=time.monotonic();save();budget.remaining()
        for actor in reversed(owner.actors):
            row=dict(actor_id=actor.id,start_wall_s=time.monotonic());report['actor_destruction'].append(row);save()
            try:row['destroyed']=bool(budget.call(actor.destroy,client=owner.client))
            except Exception as exc:row.update(destroyed=False,error=repr(exc));report['errors'].append('ActorDestroy')
            row['end_wall_s']=time.monotonic();save();budget.remaining()
        if advance_unknown:
            report['settings_status']='NotAttempted_UnknownPriorServerAdvance';save();return report
        if owner.original is None:report['settings_status']='NoOriginalSettings';save();return report
        budget.call(owner.world.set_weather,owner.weather,client=owner.client)
        before=budget.call(owner.world.get_snapshot,client=owner.client).frame
        report['settings_restore_start_wall_s']=time.monotonic();save()
        budget.call(owner.world.apply_settings,owner.original,client=owner.client,timeout_argument=True)
        actual=budget.call(owner.world.get_settings,client=owner.client)
        fields=('synchronous_mode','fixed_delta_seconds','no_rendering_mode','substepping','max_substep_delta_time','max_substeps')
        report['settings']={k:getattr(actual,k) for k in fields};report['settings_expected']={k:getattr(owner.original,k) for k in fields}
        # Read a new asynchronous snapshot so destroyed actors are not checked
        # against the pre-destruction cached frame. This waits; it does not tick.
        snap=budget.call(owner.world.wait_for_tick,client=owner.client,timeout_argument=True)
        report['settings_observed_frame_delta']=snap.frame-before
        report['remaining']=[a.id for a in owner.actors if snap.find(a.id)]
        report['settings_restored']=report['settings']==report['settings_expected']
        report['settings_restore_end_wall_s']=time.monotonic();save()
    except Exception as exc:report['errors'].append(repr(exc));save()
    report['complete']=bool(report['settings_restored'] and report['remaining']==[] and
        all(x.get('passed') for x in report['sensor_stops']) and all(x.get('destroyed') for x in report['actor_destruction']) and not report['errors'])
    save();return report
