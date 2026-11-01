"""Explicit 0.9.16 compatibility/shadow and physical test-command stops.

Only this worker ticks. Model process is network-isolated and exposes no
control payload to the owner. All vehicle commands have non-model sources.
"""
import argparse
from collections import defaultdict,Counter
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import Queue,Empty,Full
import subprocess
import sys
import time
import traceback
import numpy as np
from tools.m8j13_common import config,atomic,write,append,digest,quantiles
from tools.m8j13_sync_diagnostic import native_camera_spec,settings_record,snapshot_record,CAMERAS
from simulator.m8j13_candidate_audit import CandidateInbox
from simulator.m8j13_sensor_audit import check_group
from simulator.m8j13_safety import SafetyGate,watchdog_reason
from simulator.m8j13_carla import measurement
from datasets.m8j12_contract import official_camera_contract,native_target,measured_navigation_target

CFG=Path('configs/m8j13_compatibility.json')
STOP=dict(steer=0.,throttle=0.,brake=1.)


def nav_geometry(world,ego_tf,target):
    """Exercise the versioned HS route boundary and actual native consumer."""
    import carla
    from simulator.m8j13_navigation import geometry_checks
    source=Path(config()['runtime_root'])/'official-source/leaderboard/team_code/planner.py'
    # This owner explicitly connects only the 0.9.16 candidate. The reusable
    # route adapter has a separate preserved 0.9.15 path tested offline.
    return geometry_checks(world.get_map(),source,'0.9.16',
        anchors=((0.,0.),(ego_tf.location.x,ego_tf.location.y)))


class Owner:
    def __init__(self,root,cfg):
        import carla
        self.root=root;self.cfg=cfg;self.c=config();self.actors=[];self.sensors=[];self.tick_count=0;self.last=None;self.clients=[]
        self.client=carla.Client('127.0.0.1',self.c['port']);self.client.set_timeout(45.);self.world=self.client.get_world()
        self.original=self.world.get_settings();self.weather=self.world.get_weather()
        self.identity('owner')
        if self.original.synchronous_mode:raise RuntimeError('WorldAlreadySynchronous')
        if self.world.get_map().name.split('/')[-1]!=self.c['map']:
            self.world=self.client.load_world(self.c['map']);self.original=self.world.get_settings();self.weather=self.world.get_weather()
        self.event('initial_world',settings=settings_record(self.world),episode=self.world.id)
        s=self.world.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;s.no_rendering_mode=False
        s.substepping=True;s.max_substeps=10;s.max_substep_delta_time=.01
        self.world.apply_settings(s);self.client.set_timeout(5.)
        self.event('settings_applied',settings=settings_record(self.world),internal_tick_charge=30)
        assert settings_record(self.world)==dict(synchronous_mode=True,fixed_delta_seconds=.05,no_rendering_mode=False,substepping=True,max_substep_delta_time=.01,max_substeps=10)
        self.world.set_weather(carla.WeatherParameters.ClearNoon);self.last=self.world.get_snapshot().frame
        route=json.loads(Path(self.c['scenes'][0]['route_file']).read_text());i=self.c['scenes'][0]['start_index'];xyz=route['route_world_carla'][i]
        self.spawn_tf=carla.Transform(carla.Location(x=xyz[0],y=xyz[1],z=xyz[2]+.5),carla.Rotation(yaw=float(-np.degrees(route['route_world_hs'][i][2]))))
        self.target=self.spawn_tf.transform(carla.Location(x=cfg['fixed_target_ahead_m']))
        others=list(self.world.get_actors().filter('vehicle.*'))+list(self.world.get_actors().filter('walker.*'))
        if others:raise RuntimeError('TestAreaHasOtherTrafficParticipants')
        write(root/'geometry.json',nav_geometry(self.world,self.spawn_tf,self.target))
        self.next_cycle=time.monotonic()
    def identity(self,role):
        import carla
        assert self.client.get_client_version()==self.client.get_server_version()==self.cfg['version']
        row=dict(role=role,python=sys.executable,carla_path=carla.libcarla.__file__,client_version=self.client.get_client_version(),server_version=self.client.get_server_version(),port=self.c['port'],episode=self.world.id)
        self.clients.append(row);append(self.root/'clients.jsonl',row)
    def event(self,kind,**kw):append(self.root/'lifecycle.jsonl',dict(event=kind,wall_s=time.monotonic(),**kw))
    def spawn(self,bp,tf,parent=None):
        import carla
        args={} if parent is None else dict(attach_to=parent,attachment_type=carla.AttachmentType.Rigid)
        a=self.world.spawn_actor(bp,tf,**args);self.actors.append(a)
        if a.type_id.startswith('sensor.'):self.sensors.append(a)
        self.event('spawn',actor=a.id,type=a.type_id);return a
    def ego(self):
        return self.spawn(self.world.get_blueprint_library().find(self.c['vehicle']),self.spawn_tf)
    def apply(self,ego,control,source,folder):
        import carla
        if source not in ('IndependentTestControl','SafetyStop','LatePlanRejectedStop'):raise RuntimeError('ForbiddenControlSource')
        start=time.monotonic()
        try:ego.apply_control(carla.VehicleControl(**control))
        except Exception as exc:
            append(folder/'actuation.jsonl',dict(source=source,status='CommandNotDelivered',error=repr(exc),wall_s=start));raise
        append(folder/'actuation.jsonl',dict(source=source,command=control,call_start=start,call_end=time.monotonic(),
            issued_monotonic_s=start,valid_until_monotonic_s=start+self.cfg['max_command_age_wall_s'],model_control=False))
    def tick(self,ego,phase,folder):
        if self.tick_count>=self.cfg['run_tick_cap']:raise RuntimeError('ActiveTickBudgetExhausted')
        time.sleep(max(0.,self.next_cycle-time.monotonic()));self.next_cycle=time.monotonic()+.05
        start=time.monotonic();append(self.root/'tick_requests.jsonl',dict(index=self.tick_count,phase=phase,wall_s=start));self.tick_count+=1
        frame=self.world.tick(5.);snap=self.world.get_snapshot()
        if frame!=snap.frame or frame!=self.last+1:raise RuntimeError('CompetingTickOwner')
        self.last=frame;m=measurement(snap,ego);m.update(input_wall_s=start,tick_end_wall_s=time.monotonic(),delta_s=snap.timestamp.delta_seconds)
        u=ego.get_control();m['applied']=dict(steer=u.steer,throttle=u.throttle,brake=u.brake)
        append(folder/'snapshots.jsonl',m);return snap,m
    def destroy_owned(self):
        for a in self.sensors:
            try:
                before=a.is_listening();a.stop();after=a.is_listening();self.event('listener_stop',id=a.id,before=before,after=after)
                if after:raise RuntimeError('SensorStillListening')
            except Exception as exc:self.event('cleanup_error',error=repr(exc))
        for a in reversed(self.actors):self.event('destroy',id=a.id,result=a.destroy())
        self.actors=[];self.sensors=[]
    def close(self):
        ids=[a.id for a in self.actors];self.destroy_owned();self.world.set_weather(self.weather);self.world.apply_settings(self.original)
        fresh=self.world.wait_for_tick(5.);r=dict(settings=settings_record(self.world),remaining=[i for i in ids if fresh.find(i)],traffic_manager='NotCreated')
        self.event('restored',**r);return r


class Inputs:
    def __init__(self,owner,ego,folder):
        import carla
        self.owner=owner;self.ego=ego;self.folder=folder;self.groups=defaultdict(list);self.states={};self.payloads={};self.imu={};self.imu_queue=Queue(100);self.collision=Queue(100);self.seen=set();self.sensor_map={};self.collisions=[]
        self.identity=(folder.name,owner.world.id,folder.name+':rgb:1');self.inbox=CandidateInbox(self.identity)
        self.contract=official_camera_contract(Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
        for name in CAMERAS:
            s=self.contract['sensors'][name];bp=owner.world.get_blueprint_library().find('sensor.camera.rgb')
            for k,v in [('image_size_x',704),('image_size_y',396),('fov',s['fov']),('sensor_tick',0.)]:bp.set_attribute(k,str(v))
            tf=carla.Transform(carla.Location(x=s['x'],y=s['y'],z=s['z']),carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll']))
            a=owner.spawn(bp,tf,ego);self.sensor_map[name]=a;a.listen(self.inbox.callback(name,a.id))
            if not a.is_listening():raise RuntimeError('ListenerNotActive')
        bp=owner.world.get_blueprint_library().find('sensor.other.imu');bp.set_attribute('sensor_tick','0.0')
        owner.spawn(bp,carla.Transform(carla.Location(x=-1.4)),ego).listen(self.imu_queue.put_nowait)
        owner.spawn(owner.world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),ego).listen(self.collision.put_nowait)
    def snapshot(self,snap,m):
        self.states[m['frame']]=dict(m)
    def drain(self):
        for payload in self.inbox.drain():
            s=payload.stamp;r=s.record();key=(s.camera,s.frame)
            append(self.folder/'camera_metadata.jsonl',r)
            if key in self.seen:raise RuntimeError('DuplicateRawObservation')
            self.seen.add(key);self.groups[s.frame].append(r);self.payloads[key]=payload.bgra
        while not self.imu_queue.empty():
            x=self.imu_queue.get_nowait();self.imu[x.frame]=dict(frame=x.frame,timestamp=x.timestamp,values=[x.accelerometer.x,x.accelerometer.y,x.accelerometer.z,x.gyroscope.x,x.gyroscope.y,x.gyroscope.z,x.compass])
        while not self.collision.empty():
            x=self.collision.get_nowait();r=dict(frame=x.frame,other=x.other_actor.type_id);self.collisions.append(r);append(self.folder/'collisions.jsonl',r)
        if self.inbox.overflow:raise RuntimeError('InputQueueOverflow')
    def row(self,frame,now):
        m=self.states[frame];g=self.groups[frame]
        if len(g)!=6 or frame not in self.imu:return None,'IncompleteInput'
        check=check_group(g,dict(frame=frame,elapsed_seconds=m['sim_s']),self.identity,{n:a.id for n,a in self.sensor_map.items()})
        if not check['valid']:raise RuntimeError('InvalidInputMetadata:'+str(check['errors']))
        if now-m['input_wall_s']>self.owner.cfg['input_wait_wall_s']:return None,'InputWaitExpired'
        if self.states[max(self.states)]['sim_s']-m['sim_s']>.3+1e-6:return None,'ObservationSimulationExpired'
        imu=self.imu[frame]
        if abs(imu['timestamp']-m['sim_s'])>1e-5:raise RuntimeError('IMUTimestampMismatch')
        target=self.owner.target
        row=dict(frame_id=frame,timestamp_s=m['sim_s'],sequence_id=self.folder.name,ego_world_matrix=m['pose'],speed_mps=m['speed'],
            imu=imu['values'],imu_frame=frame,imu_timestamp_s=imu['timestamp'],navigation_command=4,target_world_xy=[target.x,target.y],
            ego_state_source='Oracle Ego State',input_capture_wall_s=m['input_wall_s'],camera_frames={r['camera']:r['frame'] for r in g},
            camera_timestamps_s={r['camera']:r['timestamp_s'] for r in g},camera_world_matrices={r['camera']:r['transform'] for r in g})
        measured=measured_navigation_target(row);geometry=native_target(m['pose'],row['target_world_xy'])
        if not np.allclose(measured,geometry,atol=.03):raise RuntimeError('MeasuredNavigationDirectionConflict')
        # Reuse official calibration closure, independently check live sensor
        # world transforms against same-frame ego * configured relative mount.
        import carla
        maximum=0.
        for r in g:
            s=self.contract['sensors'][r['camera']];tf=carla.Transform(carla.Location(x=s['x'],y=s['y'],z=s['z']),carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll']))
            maximum=max(maximum,float(np.max(np.abs(np.asarray(m['pose'])@np.asarray(tf.get_matrix())-np.asarray(r['transform'])))))
        if maximum>self.owner.cfg['pose_tolerance']:raise RuntimeError('CameraExtrinsicsMismatch')
        append(self.folder/'input_checks.jsonl',dict(frame=frame,extrinsic_max_error=maximum,navigation_error=float(np.max(np.abs(measured-geometry)))))
        return row,None
    def prune(self,frame):
        for d in [self.groups,self.states,self.imu]:
            for f in list(d):
                if f<frame-12:d.pop(f,None)
        for key in list(self.payloads):
            if key[1]<frame-12:self.payloads.pop(key,None)
    def close(self):self.inbox.close()


def shadow(owner,root):
    if json.load(open(root/'geometry.json'))['status']!='Passed':raise RuntimeError('ShadowBlockedByGPSCompatibility')
    cfg=owner.cfg;ego=owner.ego();inputs=Inputs(owner,ego,root);proc=None
    for d in ['requests','responses','native','diagnostics']:(root/d).mkdir()
    try:
        owner.apply(ego,STOP,'IndependentTestControl',root)
        with (root/'model.log').open('x') as f:
            proc=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['model'],'bash','tools/m8j12_env.sh','-m','tools.m8j13_shadow_service','--root',str(root)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        # World remains synchronous and parked; loading does not spend ticks.
        until=time.monotonic()+cfg['model_startup_wall_s']
        while not (root/'model_ready.json').exists():
            if proc.poll() is not None or (root/'model_error.json').exists():raise RuntimeError('ModelLoadFailed')
            if time.monotonic()>until:raise TimeoutError('ModelReadyTimeout')
            time.sleep(.05)
        requests=0;selected=set();responses=0;skips=Counter();first=None
        for index in range(cfg['shadow_max_ticks']):
            owner.apply(ego,STOP,'IndependentTestControl',root);snap,m=owner.tick(ego,'shadow',root);inputs.snapshot(snap,m)
            atomic(root/'current_snapshot.json',dict(frame=m['frame'],sim_s=m['sim_s']))
            inputs.drain();first=m['frame'] if first is None else first
            if inputs.collisions:raise RuntimeError('ShadowCollision')
            if abs(m['speed'])>.1 and index>=cfg['shadow_warmup_ticks']:raise RuntimeError('ShadowEgoMoving')
            files=list((root/'responses').glob('*.json'));responses=len(files)
            complete=sum(json.loads(p.read_text()).get('model_forward',False) for p in files)
            if (root/'model_error.json').exists():raise RuntimeError('ShadowModelError')
            if complete>=cfg['planned_forwards']:break
            for frame in sorted(inputs.states):
                if frame in selected or frame-first<cfg['shadow_warmup_ticks'] or (frame-first-cfg['shadow_warmup_ticks'])%cfg['shadow_request_cadence_ticks']:continue
                if len(inputs.groups[frame])<6 and time.monotonic()-inputs.states[frame]['input_wall_s']<cfg['input_wait_wall_s']:continue
                selected.add(frame)
                if requests>responses:skips['WorkerBusy']+=1;continue
                row,reason=inputs.row(frame,time.monotonic())
                if reason:skips[reason]+=1;append(root/'skips.jsonl',dict(frame=frame,reason=reason));continue
                images={n:np.frombuffer(inputs.payloads[(n,frame)],np.uint8).reshape(396,704,4)[:,:,:3].copy() for n in CAMERAS}
                name='%06d'%requests;np.savez(root/'requests'/(name+'.npz'),**images);atomic(root/'requests'/(name+'.json'),row);requests+=1
            inputs.prune(m['frame'])
        (root/'model_stop').touch();proc.wait(timeout=30)
        return dict(status='Recorded',requests=requests,skips=dict(skips),model=json.load(open(root/'model_final.json')),model_controls=0)
    finally:
        inputs.close();(root/'model_stop').touch(exist_ok=True)
        from tools.m8j13_supervisor import terminate
        terminate(proc)


def stopping(owner,root):
    cfg=owner.cfg;rounds=[]
    for case in cfg['fault_cases']:
        folder=root/case;folder.mkdir();append(root/'rounds.jsonl',dict(case=case,event='ExplicitNewTestRound',wall_s=time.monotonic()))
        ego=owner.ego();inputs=Inputs(owner,ego,folder);stub=None;wd=None;gate=SafetyGate(dict(owner.c,max_speed_mps=2.,speed_cap_brake_threshold_mps=1.7))
        injected=None;first_fault=None;held=0;rows=[];late=False;result=dict(case=case,status='Partial',model_controls=0)
        try:
            with (folder/'stub.log').open('x') as f:stub=subprocess.Popen([sys.executable,'-m','tools.m8j13_stop_planner_stub','--root',str(folder)],stdout=f,stderr=subprocess.STDOUT)
            until=time.monotonic()+5
            while not (folder/'heartbeat.json').exists():
                if time.monotonic()>until:raise TimeoutError('StubStartupTimeout')
                time.sleep(.01)
            with (folder/'watchdog.log').open('x') as f:wd=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['watchdog'],sys.executable,'-m','tools.m8j13_watchdog','--root',str(folder),'--ego',str(ego.id),'--expected-version',cfg['version']],stdout=f,stderr=subprocess.STDOUT)
            until=time.monotonic()+5
            while not (folder/'watchdog_ready.json').exists():
                if wd.poll() is not None:raise RuntimeError('WatchdogStartupFailed')
                if time.monotonic()>until:raise TimeoutError('WatchdogReadyTimeout')
                time.sleep(.01)
            owner.apply(ego,STOP,'IndependentTestControl',folder);m=None;injection_frame=None;first_round_sim=None;start_position=None
            for index in range(cfg['round_max_ticks']):
                now=time.monotonic();reason=None;raw=dict(steer=0.,throttle=cfg['test_throttle'],brake=0.)
                if index<cfg['round_settle_ticks']:raw=STOP.copy()
                if index==cfg['round_settle_ticks']:(folder/'ARM').touch()
                if m is not None and index>=cfg['round_settle_ticks'] and abs(m['speed'])>cfg['round_speed_cap_mps']:reason='TestSpeedLimitExceeded'
                if m is not None and injected is None and index>=cfg['round_settle_ticks'] and m['speed']>=cfg['fault_injection_speed_mps']:
                    injected=dict(wall_s=now,sim_s=m['sim_s'],frame=m['frame'],speed=m['speed'],position=m['position']);injection_frame=m['frame']+1
                    append(folder/'fault_chain.jsonl',dict(event='fault_injected',case=case,**injected))
                    if case=='camera_missing':inputs.sensor_map['CAM_FRONT'].stop()
                    elif case=='worker_blocked':(folder/'block_response').touch()
                    elif case=='nonfinite_control':raw=dict(steer=0.,throttle=float('nan'),brake=0.)
                    elif case=='manual_estop':(folder/'EMERGENCY_STOP').touch()
                if injected and case=='camera_missing':
                    missing=inputs.states.get(injection_frame)
                    if missing and now-missing['input_wall_s']>cfg['input_wait_wall_s']:reason='InputWaitExpired'
                fault_path=folder/'watchdog_fault.json'
                if fault_path.exists():reason=json.load(open(fault_path))['reason']
                if wd.poll() is not None:reason='WatchdogProcessExited'
                if reason:gate.stop(reason,0 if m is None else m['sim_s'],now)
                previous=gate.reason;u=gate.command(raw,speed=0 if m is None else m['speed'],sim_s=0 if m is None else m['sim_s'],wall_s=now)
                if gate.reason and first_fault is None:
                    first_fault=dict(reason=gate.reason,wall_s=now,sim_s=m['sim_s'],frame=m['frame']);append(folder/'fault_chain.jsonl',dict(event='normal_command_invalidated_stop_generated',command=u,**first_fault))
                if gate.reason and (folder/'late_response.json').exists() and not late:
                    stale=json.load(open(folder/'late_response.json'));u=gate.command(stale,speed=m['speed'],sim_s=m['sim_s'],wall_s=now)
                    if u!=STOP:raise RuntimeError('LatePlanReleasedStopLatch')
                    late=True;append(folder/'fault_chain.jsonl',dict(event='late_plan_rejected',wall_s=now,command=u))
                # Blocked planning cannot keep renewing throttle. Let the
                # independent watchdog deliver the first brake in that case.
                suppress=injected is not None and case in ('worker_blocked','manual_estop')
                if not suppress:owner.apply(ego,u,'SafetyStop' if gate.reason else 'IndependentTestControl',folder)
                elif gate.reason and late:owner.apply(ego,u,'LatePlanRejectedStop',folder)
                snap,m=owner.tick(ego,case,folder);inputs.snapshot(snap,m);inputs.drain()
                if first_round_sim is None:first_round_sim=m['sim_s']
                if index>=cfg['round_settle_ticks'] and m['speed_norm']>2.0+1e-5:gate.stop('TestSpeedLimitExceeded',m['sim_s'],time.monotonic())
                if inputs.collisions:raise RuntimeError('TestCollision')
                r=dict(index=index,measurement=m,latched_reason=gate.reason,injected=injected is not None);rows.append(r)
                stopped=gate.reason and m['speed_norm']<=cfg['stop_speed_mps'] and m['applied']['brake']>0 and m['applied']['throttle']==0
                held=held+1 if stopped else 0
                inputs.prune(m['frame'])
                if held>=math.ceil(cfg['stop_hold_sim_s']/.05)+1:
                    # Also test that a late command cannot unlock every latch.
                    check=gate.command(dict(steer=0.,throttle=.25,brake=0.),speed=m['speed'],sim_s=m['sim_s'],wall_s=time.monotonic())
                    assert check==STOP
                    result['late_command_rejected']=True;break
            post=[r['measurement'] for r in rows if first_fault and r['measurement']['sim_s']>=first_fault['sim_s']]
            brake=next((x for x in post if x['applied']['brake']>0 and x['applied']['throttle']==0),None)
            stopped=next((x for x in post if x['speed_norm']<=cfg['stop_speed_mps']),None)
            terminal=rows[-1]['measurement'];speed_ok=max(r['measurement']['speed_norm'] for r in rows[cfg['round_settle_ticks']:])<=2.+1e-5
            result.update(injected=injected,detected=first_fault,brake_observed=brake is not None,
                stopping_time_sim_s=None if stopped is None or injected is None else stopped['sim_s']-injected['sim_s'],
                stopping_distance_m=None if stopped is None or injected is None else float(np.linalg.norm(np.array(stopped['position'][:2])-injected['position'][:2])),
                terminal_speed_mps=terminal['speed_norm'],stop_hold_sim_s=max(0,held-1)*.05,
                speed_cap_respected=speed_ok,collisions=len(inputs.collisions),ticks=len(rows),
                actual_first_brake_frame=None if brake is None else brake['frame'],watchdog_independent=(folder/'watchdog_fault.json').exists(),
                real_late_worker_response_rejected=late,model_controls=0)
            ok=bool(injected and first_fault and brake and held>=21 and speed_ok and not inputs.collisions)
            result['status']='PhysicalStopObserved' if ok else 'Partial'
        except BaseException as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
        finally:
            # Keep physics and brake through observed hold; teardown is never
            # used to claim stop. New round means new actors + explicit reset.
            inputs.close();(folder/'stub_stop').touch();(folder/'watchdog_stop').touch()
            for proc in [stub,wd]:
                if proc is not None:
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
            owner.destroy_owned();write(folder/'result.json',result);rounds.append(result)
        if result['status']!='PhysicalStopObserved':break
    return dict(rounds=rounds,model_controls=0)


def worker(root,phase):
    cfg=json.loads(CFG.read_text());cfg['run_tick_cap']=cfg['shadow_max_ticks'] if phase=='shadow' else cfg['round_max_ticks']*4
    owner=None;result=dict(phase=phase,status='Partial',model_controls=0)
    try:
        # Keep a reference even if initialization raises, so settings restoration
        # can run after a geometry or interface failure.
        owner=Owner.__new__(Owner);owner.__init__(root,cfg)
        result['work']=shadow(owner,root) if phase=='shadow' else stopping(owner,root);result['status']='Recorded'
    except BaseException as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if owner is not None:
            result['explicit_ticks']=getattr(owner,'tick_count',0);result['charged_ticks']=result['explicit_ticks']+30
            try:result['cleanup']=owner.close()
            except Exception as exc:result['cleanup_error']=repr(exc)
        write(root/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def supervise(phase):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    c=config();cfg=json.loads(CFG.read_text());base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    reserve=30+(cfg['shadow_max_ticks'] if phase=='shadow' else cfg['round_max_ticks']*4)
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger_path=base/'budget.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'runs':[]}
        if len(ledger['runs'])>=2 or sum(r['reserved_ticks'] for r in ledger['runs'])+reserve>1600:raise RuntimeError('RunBudgetExhausted')
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        root=base/phase;root.mkdir();write(root/'request.json',dict(phase=phase,reserved_ticks=reserve,config=json.loads(CFG.read_text()),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hypothesis='Explicit interface/shadow compatibility' if phase=='shadow' else 'Four independent test-control faults produce actual brake and physical held stop; no model commands'))
        ledger['runs'].append(dict(phase=phase,reserved_ticks=reserve));atomic(ledger_path,ledger)
        server=None;child=None;report={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            cmd=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
            with (root/'server.log').open('x') as f:server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+150
            while time.monotonic()<until:
                if server.poll() is not None:raise RuntimeError('ServerEarlyExit')
                client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                try:
                    assert client.get_client_version()==client.get_server_version()==cfg['version'];world=client.get_world()
                    report['supervisor_client']=dict(python=sys.executable,path=carla.libcarla.__file__,port=c['port'],version=cfg['version'],episode=world.id);break
                except RuntimeError:time.sleep(1.)
            else:raise TimeoutError('ServerReadiness')
            cmd=['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_compatibility','--worker-root',str(root),'--phase',phase]
            with (root/'worker.log').open('x') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);until=time.monotonic()+900
                while child.poll() is None:
                    if server.poll() is not None:raise RuntimeError('ServerExitDuringRun')
                    if time.monotonic()>until:raise TimeoutError('WorkerDeadline')
                    time.sleep(.2)
            report['worker_exit_code']=child.returncode
            if child.returncode:raise RuntimeError('WorkerFailed')
            report['result']=json.load(open(root/'result.json'))
        except BaseException as exc:report['error']=repr(exc)
        finally:
            terminate(child);terminate(server)
            until=time.monotonic()+5.
            while time.monotonic()<until and (any(listening(p) for p in range(c['port'],c['port']+3)) or (server is not None and group_members(server.pid))):
                time.sleep(.1)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)};report['groups_after']=[group_members(p.pid) for p in [server,child] if p is not None]
            ledger['runs'][-1]['charged_ticks']=report.get('result',{}).get('charged_ticks',reserve);atomic(ledger_path,ledger)
            write(root/'supervisor.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['shadow','stopping'],required=True);p.add_argument('--worker-root',type=Path);a=p.parse_args();os.umask(0o077)
    if a.worker_root:write(a.worker_root/'worker_started.json',dict(pid=os.getpid()));worker(a.worker_root,a.phase)
    else:supervise(a.phase)


if __name__=='__main__':main()
