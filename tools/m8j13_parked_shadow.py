"""Single-start parked live shadow. Only constant independent brake can actuate."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from tools.m8j13_compatibility import Owner,Inputs,STOP
from tools.m8j13_common import config,write,atomic,append,digest
from simulator.m8j13_navigation import load_planner,set_navigation_route
from simulator.m8j13_shadow_contract import require_parked_command,require_completion_only,candidate_offset,parked_measurement_valid
from simulator.m8j13_execution import control_record,feedback_matches
from simulator.m8j13_safety import PlanStamp,freshness
from simulator.m8j13_transport import Slots

CFG=Path('configs/m8j13_parked_shadow.json')


def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default


class ParkedOwner(Owner):
    def apply(self,ego,control,source,folder):
        require_parked_command(control)
        if source!='IndependentParkedBrake':raise RuntimeError('ParkedControlSourceViolation')
        # Base class emits only numeric command data, never model IPC.
        super().apply(ego,STOP,'SafetyStop',folder)
        append(folder/'parked_control_sources.jsonl',dict(source=source,wall_s=time.monotonic(),model_control=False))
    def brake(self,ego):self.apply(ego,STOP,'IndependentParkedBrake',self.root)
    def parked_tick(self,ego,inputs,phase,state):
        self.brake(ego);issued=time.monotonic()
        state['issued']=dict(issued_wall_s=issued,valid_until_wall_s=issued+.3)
        atomic(self.root/'execution_state.json',state)
        snap,m=self.tick(ego,phase,self.root);full=control_record(ego.get_control())
        request=dict(STOP,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
        matched=feedback_matches(request,full)
        append(self.root/'parked_feedback.jsonl',dict(frame=m['frame'],sim_s=m['sim_s'],wall_s=time.monotonic(),speed_mps=m['speed_norm'],control=full,matched=matched,phase=phase))
        if not matched:raise RuntimeError('ParkedBrakeFeedbackMismatch')
        state['ack']=dict(confirmed_wall_s=time.monotonic(),matched=matched)
        atomic(self.root/'execution_state.json',state)
        inputs.snapshot(snap,m);atomic(self.root/'current_snapshot.json',dict(frame=m['frame'],sim_s=m['sim_s']))
        if self.cfg.get('per_camera'):atomic(self.root/'tick_pending.json',dict(pending=False,frame=m['frame'],wall_s=time.monotonic()))
        begin=time.monotonic();inputs.drain();end=time.monotonic()
        append(self.root/'owner_timing.jsonl',dict(frame=m['frame'],phase=phase,drain_and_metadata_write_ms=(end-begin)*1000,
            tick_wall_ms=(m['tick_end_wall_s']-m['input_wall_s'])*1000))
        if inputs.collisions:raise RuntimeError('ParkedCollision')
        return m


def worker(root):
    cfg=json.loads(CFG.read_text());cfg['run_tick_cap']=cfg['max_active_ticks']-cfg['settings_tick_charge']
    owner=None;inputs=None;ego=None;model=None;watchdog=None;result=dict(status='Partial',model_controls=0,training=0)
    slots=None;request_rows={};relay=None;early_responses=set();camera_schedule={}
    state=dict(armed=False,require_planner=False,issued=None,ack=None);maximum=0.;measured=[];pending=[];decided=set();responses=set();requests=0;last=None;exit_ticks=0
    try:
        owner=ParkedOwner.__new__(ParkedOwner);owner.__init__(root,cfg)
        ego=owner.ego();owner.brake(ego);inputs=Inputs(owner,ego,root)
        source=Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/planner.py';map_=owner.world.get_map()
        inputs.navigation=load_planner(source,map_);gps=map_.transform_to_geolocation(owner.target)
        mode=set_navigation_route(inputs.navigation,[({'lat':gps.latitude,'lon':gps.longitude,'alt':gps.altitude},SimpleNamespace(value=4))],
            coordinate_type='gps',client_version=cfg['version'],carla_map=map_)
        write(root/'navigation_wiring.json',dict(mode=mode,consumer='Inputs.row -> original RoutePlanner.run_step -> measured_navigation_target',source_sha256=digest(source)))
        for name in ('requests','responses','native','diagnostics','input_rows'):(root/name).mkdir()
        if cfg.get('pipeline_service'):
            if cfg.get('per_camera'):
                from simulator.m8j13_partial_transport import PartialSlots
                slots=PartialSlots(Path('/dev/shm')/('hs-j13-'+str(os.getpid())),create=True)
            else:slots=Slots(Path('/dev/shm')/('hs-j13-'+str(os.getpid())),create=True)
            atomic(root/'transport.json',dict(root=str(slots.root),capacity=2,shape=[6,396,704,4],dtype='uint8'))
        if cfg.get('per_camera'):
            from tools.m8j13_camera_live import EarlyRelay
            identity=dict(run=inputs.identity[0],episode=inputs.identity[1],generation=inputs.identity[2],sensors={n:a.id for n,a in inputs.sensor_map.items()})
            relay=EarlyRelay(inputs.inbox,slots,identity,root);inputs.inbox=relay
            def before_tick(frame,wall,phase):
                if frame in camera_schedule:relay.reserve_frame(frame,wall,camera_schedule[frame])
                atomic(root/'tick_pending.json',dict(pending=True,frame=frame,wall_s=wall))
            owner.before_tick=before_tick
        atomic(root/'execution_state.json',state)
        with (root/'watchdog.log').open('x') as f:
            watchdog=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['watchdog'],sys.executable,'-m','tools.m8j13_watchdog','--root',str(root),'--ego',str(ego.id),'--expected-version',cfg['version'],'--execution-contract'],stdout=f,stderr=subprocess.STDOUT)
        until=time.monotonic()+10
        while not (root/'watchdog_ready.json').exists():
            owner.brake(ego)
            if watchdog.poll() is not None:raise RuntimeError('WatchdogStartupFailed')
            if time.monotonic()>until:raise TimeoutError('WatchdogReadiness')
            time.sleep(.05)
        with (root/'model.log').open('x') as f:
            model=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['model'],'bash','tools/m8j12_env.sh','-m',('tools.m8j13_pipeline_service' if cfg.get('pipeline_service') else 'tools.m8j13_shadow_service'),'--root',str(root),'--run-config',str(CFG)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(root/'model_process.json',dict(pid=model.pid,pgid=model.pid))
        until=time.monotonic()+cfg['model_startup_wall_s']
        while not (root/'model_ready.json').exists():
            owner.brake(ego)  # synchronous world not advanced during loading
            if model.poll() is not None or (root/'model_error.json').exists():raise RuntimeError('ModelLoadFailed')
            if time.monotonic()>until:raise TimeoutError('ModelLoadDeadline')
            time.sleep(.05)
        ready=load(root/'model_ready.json')
        if ready['network_namespace']==os.readlink('/proc/self/ns/net'):raise RuntimeError('ModelNetworkNotIsolated')
        if not all(ready[k] for k in ('parameters_frozen','eval','fp32')):raise RuntimeError('ModelFreezeContract')
        for i in range(cfg['initialization_ticks']):
            m=owner.parked_tick(ego,inputs,'initialization',state);last=m
            measured.append(m['speed_norm']);inputs.prune(m['frame'])
        if not all(parked_measurement_valid(x) for x in measured[-cfg['stationary_hold_ticks']:]):raise RuntimeError('InitialParkedHoldFailed')
        write(root/'parked_admission.json',dict(status='ParkedShadowOnlyAdmitted',initialization_ticks=cfg['initialization_ticks'],
            hold_ticks=cfg['stationary_hold_ticks'],max_hold_speed_mps=max(measured[-cfg['stationary_hold_ticks']:]),
            moving_faults_required_for_model_driving=True,model_execution_channel=False,old_ready_false_preserved=True))
        state['armed']=True;atomic(root/'execution_state.json',state);origin=last['frame']+1
        schedule={origin+candidate_offset(i,count=cfg['candidate_count']):i for i in range(cfg['candidate_count'])}
        camera_schedule.update(schedule)
        if relay is not None:
            write(root/'first_camera_schedule.json',dict(first_frame=origin,count=cfg['candidate_count'],interval=4))
        def consume_responses():
            for path in sorted((root/'responses').glob('*.json')):
                if path.name in responses:continue
                value=load(path);require_completion_only(value)
                if value['status']=='PartialCancelled':
                    if path.name not in early_responses:relay.release(path.name);early_responses.add(path.name)
                    continue
                responses.add(path.name)
                received=time.monotonic();row=request_rows.get(path.name);extra={}
                if row is not None:
                    stamp=PlanStamp(row['sequence_id'],row['frame_id'],row['timestamp_s'],row['input_capture_wall_s'],row['input_capture_wall_s'],received,'frozen')
                    extra=dict(wall_age_s=received-row['input_capture_wall_s'],sim_age_s=last['sim_s']-row['timestamp_s'],
                        age_reason=freshness(stamp,row['sequence_id'],last['frame'],last['sim_s'],received))
                append(root/'completion_received.jsonl',dict(request=path.name,wall_s=received,owner_consume_check_done_wall_s=time.monotonic(),response=value,**extra))
                if relay is not None:relay.release(path.name)
                elif slots is not None:slots.release(path.name)
        def candidates():
            nonlocal requests
            consume_responses()
            for frame in pending:
                if frame in decided:continue
                observation=inputs.states.get(frame)
                if observation is None:raise RuntimeError('CandidateStateLostBeforeDecision')
                now=time.monotonic();complete=len(inputs.groups[frame])==6 and frame in inputs.imu
                if not complete and now-observation['input_wall_s']<cfg['input_wait_wall_s']:break
                decided.add(frame);begin=now
                row,reason=inputs.row(frame,now)
                record=dict(candidate_index=schedule[frame],frame=frame,input_wall_s=observation['input_wall_s'],input_sim_s=observation['sim_s'],
                    decision_wall_s=now,sim_age_s=last['sim_s']-observation['sim_s'],camera_arrival_wall_s={x['camera']:x['receive_monotonic_s'] for x in inputs.groups[frame]},
                    camera_copy_done_wall_s={x['camera']:x['copied_monotonic_s'] for x in inputs.groups[frame]})
                if reason:
                    if relay is not None:relay.reject(frame)
                    record.update(status='Rejected',reason=reason,group_checks_ms=(time.monotonic()-begin)*1000);append(root/'candidates.jsonl',record);continue
                if (relay is not None and relay.frames[frame]['status']=='QueueCapacity') or (relay is None and requests-len(responses)>=(2 if slots is not None else 1)):
                    record.update(status='Skipped',reason='QueueCapacity' if slots is not None else 'WorkerBusy',group_checks_ms=(time.monotonic()-begin)*1000);append(root/'candidates.jsonl',record);continue
                group_done=time.monotonic()
                name='%06d.json'%schedule[frame]
                if relay is not None:
                    saved=time.monotonic()
                    try:name=relay.commit(frame,row,observation,saved,last['sim_s'])
                    except Exception as exc:
                        record.update(status='Rejected',reason='PartialCommitError:'+repr(exc));append(root/'candidates.jsonl',record);raise
                    request_rows[name]=row;requests+=1
                    record.update(status='Submitted',request=name,submitted_wall_s=saved,group_checks_ms=(saved-begin)*1000,
                        image_copy_ms=0.,ipc_save_ms=0.,queue_occupied=len(slots.leases)-1)
                    append(root/'candidates.jsonl',record);continue
                if slots is not None:
                    identity=dict(run=inputs.identity[0],episode=inputs.identity[1],generation=inputs.identity[2],
                        sensors={n:a.id for n,a in inputs.sensor_map.items()})
                    token=slots.publish(name,row,{n:inputs.payloads[(n,frame)] for n in inputs.sensor_map},identity)
                    if token is None:raise RuntimeError('LeaseCapacityAccountingMismatch')
                    copied=time.monotonic();saved=copied;payload=token
                else:
                    images={n:np.frombuffer(inputs.payloads[(n,frame)],np.uint8).reshape(396,704,4)[:,:,:3].copy() for n in inputs.sensor_map}
                    copied=time.monotonic();np.savez(root/'requests'/Path(name).with_suffix('.npz'),**images);saved=time.monotonic();payload=row
                stamp=PlanStamp(row['sequence_id'],frame,row['timestamp_s'],row['input_capture_wall_s'],saved,saved,'frozen')
                reason=freshness(stamp,row['sequence_id'],last['frame'],last['sim_s'],saved)
                record.update(group_checks_ms=(group_done-begin)*1000,image_copy_ms=(copied-group_done)*1000,
                    ipc_save_ms=(saved-copied)*1000,request=name,submitted_wall_s=saved,queue_occupied=requests-len(responses))
                if reason:
                    if slots is not None:slots.release(name)
                    record.update(status='Rejected',reason=reason+'BeforeIPC');append(root/'candidates.jsonl',record);continue
                atomic(root/'requests'/name,payload);request_rows[name]=row;requests+=1;record['status']='Submitted';append(root/'candidates.jsonl',record)
        for index in range(cfg['candidate_window_ticks']+cfg['pending_drain_max_ticks']):
            m=owner.parked_tick(ego,inputs,'measurement' if index<cfg['candidate_window_ticks'] else 'pending_drain',state);last=m
            maximum=max(maximum,m['speed_norm'])
            if not parked_measurement_valid(m['speed_norm']):raise RuntimeError('ParkedEgoMoved_StopNewInputs')
            if watchdog.poll() is not None:raise RuntimeError('WatchdogExited')
            if (root/'model_error.json').exists():raise RuntimeError('ShadowModelError')
            consume_responses()
            entries=[json.loads(x) for x in (root.parent/'forward_budget.jsonl').read_text().splitlines()] if (root.parent/'forward_budget.jsonl').exists() else []
            attempts=sum(x['kind']=='live_pipeline' for x in entries) if slots is not None else len(entries)
            if m['frame'] in schedule:pending.append(m['frame'])
            if attempts>=cfg['max_forward_attempts']:
                if requests==len(responses):
                    if relay is not None:
                        for f,item in list(relay.frames.items()):
                            if item['status']=='Partial':relay.reject(f)
                    break
                # Wait within bounded ticks without admitting more observations.
            else:
                candidates()
            if model.poll() is not None and attempts<cfg['max_forward_attempts']:raise RuntimeError('ModelWorkerExited')
            if index>=cfg['candidate_window_ticks'] and len(decided)==len(pending) and requests==len(responses):break
            inputs.prune(m['frame'])
        consume_responses()
        for frame in pending:
            if frame not in decided:
                if relay is not None:relay.reject(frame)
                append(root/'candidates.jsonl',dict(candidate_index=schedule[frame],frame=frame,status='NotSubmitted',reason='RunOrForwardBudgetEnded'));decided.add(frame)
        (root/'model_stop').touch()
        until=time.monotonic()+cfg['model_exit_wall_s']
        while model.poll() is None:
            owner.brake(ego)
            if time.monotonic()>until:raise TimeoutError('ModelExitDeadline')
            time.sleep(.05)
        result.update(status='Recorded',scheduled_candidates=cfg['candidate_count'],reached_candidates=len(pending),decided_candidates=len(decided),requests=requests,
            responses=len(responses),max_measurement_speed_mps=maximum,model=load(root/'model_final.json'),model_controls=0)
    except BaseException as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        (root/'model_stop').touch(exist_ok=True)
        if owner is not None and ego is not None:
            try:
                for _ in range(min(cfg['exit_observation_ticks'],cfg['run_tick_cap']-owner.tick_count)):
                    if inputs is None:break
                    m=owner.parked_tick(ego,inputs,'exit_brake_observation',state);exit_ticks+=1
                result['exit_terminal_speed_mps']=m['speed_norm'] if exit_ticks else None
            except Exception as exc:result['exit_response']='ResponseUnverified:'+repr(exc)
        if inputs is not None:inputs.close()
        (root/'watchdog_stop').touch()
        if watchdog is not None:
            try:watchdog.wait(timeout=3)
            except subprocess.TimeoutExpired:watchdog.kill();watchdog.wait(timeout=3)
        from tools.m8j13_supervisor import terminate
        terminate(model)
        if slots is not None:
            try:slots.close();result['transport_cleanup']=True
            except Exception as exc:result['transport_cleanup_error']=repr(exc)
        if owner is not None:
            result.update(explicit_ticks=owner.tick_count,charged_ticks=owner.tick_count+30,exit_observation_ticks=exit_ticks)
            try:result['cleanup']=owner.close()
            except Exception as exc:result['cleanup_error']=repr(exc)
        write(root/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def supervise():
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    cfg=json.loads(CFG.read_text());c=config();base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    if cfg.get('pipeline_service'):
        if load(base/'equivalence_result.json',{}).get('status')!='Passed':raise RuntimeError('InputEquivalenceRequired')
        if len((base/'forward_budget.jsonl').read_text().splitlines())!=cfg['offline_forward_cap']:raise RuntimeError('OfflineBudgetMismatch')
        if cfg.get('per_camera') and load(base/'online_admission.json',{}).get('admitted') is not True:raise RuntimeError('OfflineReachabilityEvidenceRequired')
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=load(base/'budget.json',dict(runs=[]))
        if ledger['runs']:raise RuntimeError('SingleStartBudgetConsumed')
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        root=base/'live';root.mkdir();write(root/'request.json',dict(config=cfg,config_sha256=digest(CFG),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),asset_record_sha256=digest(base/'assets_before.json'),
            purpose='One parked live FP32 shadow sequence with complete timings, no model actuation',
            model_forward_budget=cfg['max_forward_attempts'],planned_tick_upper_bound=sum(cfg[k] for k in ('settings_tick_charge','initialization_ticks','candidate_window_ticks','pending_drain_max_ticks','exit_observation_ticks')),model_controls=0))
        ledger['runs'].append(dict(phase='input_pipeline' if cfg.get('pipeline_service') else 'parked_shadow',reserved_ticks=cfg['max_active_ticks']));atomic(base/'budget.json',ledger)
        server=None;child=None;report={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=cfg['python_deps']+':'+str(Path.cwd()))
        try:
            cmd=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
            with (root/'server.log').open('x') as f:server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+cfg['server_startup_wall_s']
            while time.monotonic()<until:
                if server.poll() is not None:raise RuntimeError('ServerEarlyExit')
                client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                try:
                    versions=[client.get_client_version(),client.get_server_version()]
                    if versions!=[cfg['version']]*2:raise ValueError('ClientServerVersionMismatch')
                    world=client.get_world();report['client']=dict(python=sys.executable,path=carla.libcarla.__file__,versions=versions,port=c['port'],episode=world.id);break
                except RuntimeError:time.sleep(1.)
            else:raise TimeoutError('ServerReadiness')
            with (root/'worker.log').open('x') as f:
                child=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m',cfg.get('owner_module','tools.m8j13_parked_shadow'),'--worker-root',str(root),'--run-config',str(CFG)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                until=time.monotonic()+cfg['worker_wall_s']
                while child.poll() is None:
                    if server.poll() is not None:raise RuntimeError('ServerExited')
                    if time.monotonic()>until:raise TimeoutError('OwnerWallBudgetExceeded')
                    time.sleep(.2)
            report['worker_exit_code']=child.returncode;report['result']=load(root/'result.json')
        except BaseException as exc:report['error']=repr(exc)
        finally:
            (root/'model_stop').touch(exist_ok=True)
            terminate(child)
            # The model has a separate process group: owner failure must not orphan it.
            info=load(root/'model_process.json')
            if info and group_members(info['pgid']):
                import signal
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    try:os.killpg(info['pgid'],sig)
                    except ProcessLookupError:pass
                    time.sleep(.2)
            model_members=group_members(info['pgid']) if info else []
            report['model_group_after']=model_members
            transport=load(root/'transport.json')
            if transport and not model_members and Path(transport['root']).exists():Slots(transport['root']).close()
            terminate(server);until=time.monotonic()+8
            while time.monotonic()<until and (any(listening(p) for p in range(c['port'],c['port']+3)) or (server and group_members(server.pid))):time.sleep(.1)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)};report['groups_after']=[group_members(p.pid) for p in (server,child) if p]
            report['cleanup_complete']=not any(report['ports_after'].values()) and not any(report['groups_after']) and not report['model_group_after']
            ledger['runs'][-1]['charged_ticks']=(report.get('result') or {}).get('charged_ticks',cfg['max_active_ticks']);atomic(base/'budget.json',ledger)
            write(root/'supervisor.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    global CFG
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path);p.add_argument('--run-config',type=Path,default=CFG);a=p.parse_args();CFG=a.run_config;os.umask(0o077)
    if a.worker_root:worker(a.worker_root)
    else:supervise()


if __name__=='__main__':main()
