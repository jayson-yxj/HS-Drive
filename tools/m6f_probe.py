"""Finite isolated 0916 observability experiments; never default driving code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import carla
import numpy as np
from simulator.m6f_telemetry import observe, validate_pair
from simulator.m6e_actuation import cycle_command, apply_command, serialize_physics
from tools.m47_common import physics_dict, sample_vehicle
from tools.m6f_launch import inventory, verify_loaded
from simulator.m6c_geometry import road_grade


def write(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def resource():
    return dict(processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True),
                gpu=subprocess.check_output(['nvidia-smi'],text=True))


def session(cfg):
    client=carla.Client('localhost',cfg['port']);client.set_timeout(120)
    actual=(client.get_client_version(),client.get_server_version())
    if actual!=(cfg['version'],cfg['version']):raise RuntimeError(str(actual))
    world=client.get_world()
    if world.get_settings().synchronous_mode:raise RuntimeError('World already owned')
    world=client.load_world(cfg['map'])
    original=world.get_settings();settings=world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=cfg['dt']
    settings.substepping=True;settings.max_substep_delta_time=cfg['max_substep_dt'];settings.max_substeps=cfg['max_substeps']
    world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
    return client,world,original


def trial(cfg,task,root):
    dest=root/task['name'];dest.mkdir(exist_ok=False)
    write(dest/'config.json',task)
    client,w,original=session(cfg);ego=None;original_physics=None;sensors=[];rows=[];collisions=[]
    try:
        spawn=w.get_map().get_spawn_points()[task['spawn']]
        ego=w.spawn_actor(w.get_blueprint_library().find(cfg['blueprint']),spawn);w.tick()
        original_physics=ego.get_physics_control();p0=physics_dict(ego)
        gate_path=Path('reports/m6f_execution_logs/telemetry_gate.json')
        if gate_path.exists() and sha(p0)!=json.loads(gate_path.read_text())['physics_P0_sha256']:
            raise ValueError('Fresh P0 differs from frozen 0916 baseline')
        before=dict(body=sample_vehicle(w,ego),telemetry=observe(w,ego))
        if task['physics']=='P0-sham-reapply':ego.apply_physics_control(original_physics)
        elif task['physics']=='P1-wheel-brake-experimental':
            p=ego.get_physics_control();wheels=p.wheels
            for wheel in wheels:wheel.max_brake_torque*=.5
            p.wheels=wheels;ego.apply_physics_control(p)
        after=dict(body=sample_vehicle(w,ego),telemetry=observe(w,ego))
        actual=physics_dict(ego);expected=json.loads(json.dumps(p0))
        if task['physics']=='P1-wheel-brake-experimental':
            for wheel in expected['wheels']:wheel['max_brake_torque']*=.5
        if expected!=actual:raise ValueError('Physics readback differs')
        write(dest/'physics.json',dict(name='CARLA0916_PhysicsP0' if task['physics'].startswith('P0') else 'CARLA0916_P1_NoGo_Diagnostic',
            invariant=actual,sha256=sha(actual),original_sha256=sha(p0),full=serialize_physics(ego)))
        write(dest/'initialization.json',dict(before_operation=before,after_operation=after,spawn=str(spawn),road=road_grade(w.get_map(),before['body']['position'])))
        sensor=w.spawn_actor(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id)));sensors.append(sensor)
        schedule=[]
        if task['mode']=='cycle':
            for phase,count in cfg['cycle_phases']:schedule.extend((phase,i,True) for i in range(count))
        elif task['mode']=='feasibility':
            for phase,count in [('initialize',30),('cold_run',120),('normal_stop',100),('hold_final',40)]:schedule.extend((phase,i,True) for i in range(count))
        elif task['mode']=='coast':
            for phase,count in [('initialize',30),('cold_run',120),('coast',200),('hold_final',40)]:schedule.extend((phase,i,True) for i in range(count))
        elif task['mode']=='overhead':
            schedule.extend(('initialize',i,False) for i in range(40))
            for block,on in enumerate([False,True,True,False]*2):schedule.extend(('overhead_'+str(block),i,on) for i in range(100))
        epoch=time.perf_counter();slot=0
        with (dest/'trace.jsonl').open('x') as f:
            for index,(phase,i,on) in enumerate(schedule):
                scheduled=epoch+slot*.05
                if time.perf_counter()<scheduled:time.sleep(scheduled-time.perf_counter())
                begin=time.perf_counter();pre=observe(w,ego) if on else None;body_pre=sample_vehicle(w,ego)
                if phase.startswith('overhead'):kind,u,target=cycle_command('A','hold_final',0)
                else:kind,u,target=cycle_command(task['interface'],phase,i*.05)
                t_apply=time.perf_counter();apply_command(ego,kind,u);apply_end=time.perf_counter()
                frame=w.tick();tick_end=time.perf_counter();body=sample_vehicle(w,ego);post=observe(w,ego) if on else None
                repeated=observe(w,ego) if task['mode']=='feasibility' else None
                if on:
                    validate_pair(pre,post,frame)
                    if pre['associated_frame']!=body_pre['frame']:raise ValueError('PreSnapshotMismatch')
                if body['frame']!=frame:raise ValueError('BodySnapshotMismatch')
                end=time.perf_counter();next_slot=max(slot+1,int(math.floor((end-epoch)/.05))+1)
                r=dict(source='Privileged',carla_version=cfg['version'],physics_sha256=sha(actual),phase=phase,phase_tick=i,
                    requested=u,body_before=body_pre,body=body,telemetry_before=pre,telemetry_after=post,repeated_after=repeated,
                    true_engine_wheel_api=on,timeline=dict(scheduled=scheduled,start=begin,apply=t_apply,apply_end=apply_end,tick_end=tick_end,end=end),
                    deadline_missed=end>scheduled+.05,skipped_slots=next_slot-slot-1)
                f.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r);slot=next_slot
                if collisions:break
        verify_loaded()
        write(dest/'result.json',dict(kind='actual_CARLA0916',source='Privileged',carla_version=client.get_server_version(),
            client_version=client.get_client_version(),physics_sha256=sha(actual),true_engine_wheel_api=True,
            source_commit=inventory()['git_commit'],rows=len(rows),complete=len(rows)==len(schedule),collisions=collisions,
            physics_final_sha256=sha(physics_dict(ego))))
    finally:
        for s in sensors:s.stop();s.destroy()
        if ego:
            if original_physics is not None and task['physics']!='P0-unmodified':ego.apply_physics_control(original_physics)
            ego.destroy()
        w.apply_settings(original)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('batch',choices=['feasibility','responses','sham','overhead']);args=parser.parse_args()
    cfg=json.loads(Path('configs/m6f/protocol.json').read_text());root=Path('reports/m6f_trials_0916')/args.batch;root.mkdir(parents=True,exist_ok=False)
    tasks=[]
    if args.batch=='feasibility':tasks=[dict(name='p27_api',spawn=27,mode='feasibility',physics='P0-unmodified',interface='A')]
    elif args.batch=='overhead':tasks=[dict(name='p27_abba',spawn=27,mode='overhead',physics='P0-unmodified',interface='A')]
    else:
        gate=json.loads(Path('reports/m6f_execution_logs/telemetry_gate.json').read_text())
        if not gate['response_probe_go']:raise RuntimeError('Real telemetry prerequisite failed')
        if args.batch=='responses':
            for rep in cfg['repeat_ids']:
                for pos in cfg['cycle_positions']:
                    for interface in cfg['interfaces']:tasks.append(dict(name=f'p{pos}_{interface}_{rep}',spawn=pos,mode='cycle',physics='P0-unmodified',interface=interface,repeat_id=rep))
                for pos in cfg['coast_positions']:tasks.append(dict(name=f'p{pos}_coast_{rep}',spawn=pos,mode='coast',physics='P0-unmodified',interface='A',repeat_id=rep))
        else:
            for rep in cfg['repeat_ids']:
                groups=cfg['sham_groups'] if rep==401 else list(reversed(cfg['sham_groups']))
                for group in groups:tasks.append(dict(name=f'{group}_{rep}',spawn=cfg['sham_position'],mode='cycle',physics=group,interface='A',repeat_id=rep))
    write(root/'launch.json',dict(protocol=cfg,tasks=tasks,loaded=inventory(),resources=resource()))
    for i,t in enumerate(tasks):
        verify_loaded()
        try:trial(cfg,t,root)
        except BaseException as exc:
            write(root/'failure.json',dict(task=t,error=repr(exc),loaded=inventory()))
            raise
        print(i+1,len(tasks),t['name'],flush=True)
    verify_loaded();write(root/'completion.json',dict(loaded=inventory(),resources=resource(),complete=True))

if __name__=='__main__':main()
