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
        begin=time.monotonic();inputs.drain();end=time.monotonic()
        append(self.root/'owner_timing.jsonl',dict(frame=m['frame'],phase=phase,drain_and_metadata_write_ms=(end-begin)*1000,
            tick_wall_ms=(m['tick_end_wall_s']-m['input_wall_s'])*1000))
        if inputs.collisions:raise RuntimeError('ParkedCollision')
        return m


def worker(root):
    cfg=json.loads(CFG.read_text());cfg['run_tick_cap']=cfg['max_active_ticks']-cfg['settings_tick_charge']
    owner=None;inputs=None;ego=None;model=None;watchdog=None;result=dict(status='Partial',model_controls=0,training=0)
    state=dict(armed=False,require_planner=False,issued=None,ack=None);maximum=0.;measured=[];pending=[];decided=set();responses=set();requests=0;last=None;exit_ticks=0
    try:
        owner=ParkedOwner.__new__(ParkedOwner);owner.__init__(root,cfg)
        ego=owner.ego();owner.brake(ego);inputs=Inputs(owner,ego,root)
        source=Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/planner.py';map_=owner.world.get_map()
        inputs.navigation=load_planner(source,map_);gps=map_.transform_to_geolocation(owner.target)
        mode=set_navigation_route(inputs.navigation,[({'lat':gps.latitude,'lon':gps.longitude,'alt':gps.altitude},SimpleNamespace(value=4))],
            coordinate_type='gps',client_version=cfg['version'],carla_map=map_)
        write(root/'navigation_wiring.json',dict(mode=mode,consumer='Inputs.row -> original RoutePlanner.run_step -> measured_navigation_target',source_sha256=digest(source)))
        for name in ('requests','responses','native','diagnostics'):(root/name).mkdir()
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
            model=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['model'],'bash','tools/m8j12_env.sh','-m','tools.m8j13_shadow_service','--root',str(root),'--run-config',str(CFG)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
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
        schedule={origin+candidate_offset(i):i for i in range(cfg['candidate_count'])}
        def consume_responses():
            for path in sorted((root/'responses').glob('*.json')):
                if path.name in responses:continue
                value=load(path);require_completion_only(value);responses.add(path.name)
                append(root/'completion_received.jsonl',dict(request=path.name,wall_s=time.monotonic(),response=value))
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
                    record.update(status='Rejected',reason=reason,group_checks_ms=(time.monotonic()-begin)*1000);append(root/'candidates.jsonl',record);continue
                if requests>len(responses):
                    record.update(status='Skipped',reason='WorkerBusy',group_checks_ms=(time.monotonic()-begin)*1000);append(root/'candidates.jsonl',record);continue
                group_done=time.monotonic()
                images={n:np.frombuffer(inputs.payloads[(n,frame)],np.uint8).reshape(396,704,4)[:,:,:3].copy() for n in inputs.sensor_map}
                copied=time.monotonic();name='%06d'%schedule[frame]
                np.savez(root/'requests'/(name+'.npz'),**images);saved=time.monotonic()
                stamp=PlanStamp(row['sequence_id'],frame,row['timestamp_s'],row['input_capture_wall_s'],saved,saved,'frozen')
                reason=freshness(stamp,row['sequence_id'],last['frame'],last['sim_s'],saved)
                record.update(group_checks_ms=(group_done-begin)*1000,image_copy_ms=(copied-group_done)*1000,
                    ipc_save_ms=(saved-copied)*1000,request=name+'.json',submitted_wall_s=saved)
                if reason:
                    record.update(status='Rejected',reason=reason+'BeforeIPC');append(root/'candidates.jsonl',record);continue
                atomic(root/'requests'/(name+'.json'),row);requests+=1;record['status']='Submitted';append(root/'candidates.jsonl',record)
        for index in range(cfg['candidate_window_ticks']+cfg['pending_drain_max_ticks']):
            m=owner.parked_tick(ego,inputs,'measurement' if index<cfg['candidate_window_ticks'] else 'pending_drain',state);last=m
            maximum=max(maximum,m['speed_norm'])
            if not parked_measurement_valid(m['speed_norm']):raise RuntimeError('ParkedEgoMoved_StopNewInputs')
            if watchdog.poll() is not None:raise RuntimeError('WatchdogExited')
            if (root/'model_error.json').exists():raise RuntimeError('ShadowModelError')
            consume_responses()
            attempts=len((root.parent/'forward_budget.jsonl').read_text().splitlines()) if (root.parent/'forward_budget.jsonl').exists() else 0
            if attempts>=cfg['max_forward_attempts']:
                if requests==len(responses):break
                # Wait within bounded ticks without admitting more observations.
            else:
                if m['frame'] in schedule:pending.append(m['frame'])
                candidates()
            if model.poll() is not None and attempts<cfg['max_forward_attempts']:raise RuntimeError('ModelWorkerExited')
            if index>=cfg['candidate_window_ticks'] and len(decided)==len(pending) and requests==len(responses):break
            inputs.prune(m['frame'])
        consume_responses()
        for frame in pending:
            if frame not in decided:
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
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=load(base/'budget.json',dict(runs=[]))
        if ledger['runs']:raise RuntimeError('SingleStartBudgetConsumed')
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        root=base/'live';root.mkdir();write(root/'request.json',dict(config=cfg,config_sha256=digest(CFG),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),asset_record_sha256=digest(base/'assets_before.json'),
            purpose='One parked live FP32 shadow sequence with complete timings, no model actuation',
            model_forward_budget=16,planned_tick_upper_bound=30+60+320+80+26,model_controls=0))
        ledger['runs'].append(dict(phase='parked_shadow',reserved_ticks=600));atomic(base/'budget.json',ledger)
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
                child=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_parked_shadow','--worker-root',str(root)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                until=time.monotonic()+cfg['worker_wall_s']
                while child.poll() is None:
                    if server.poll() is not None:raise RuntimeError('ServerExited')
                    if time.monotonic()>until:raise TimeoutError('OwnerWallBudgetExceeded')
                    time.sleep(.2)
            report['worker_exit_code']=child.returncode;report['result']=load(root/'result.json')
        except BaseException as exc:report['error']=repr(exc)
        finally:
            terminate(child);terminate(server);until=time.monotonic()+8
            while time.monotonic()<until and (any(listening(p) for p in range(c['port'],c['port']+3)) or (server and group_members(server.pid))):time.sleep(.1)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)};report['groups_after']=[group_members(p.pid) for p in (server,child) if p]
            report['cleanup_complete']=not any(report['ports_after'].values()) and not any(report['groups_after'])
            ledger['runs'][-1]['charged_ticks']=(report.get('result') or {}).get('charged_ticks',600);atomic(base/'budget.json',ledger)
            write(root/'supervisor.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path);a=p.parse_args();os.umask(0o077)
    if a.worker_root:worker(a.worker_root)
    else:supervise()


if __name__=='__main__':main()
