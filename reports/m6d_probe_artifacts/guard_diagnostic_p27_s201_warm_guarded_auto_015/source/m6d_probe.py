"""Finite hold/release probes, original Physics; no planner or wall-clock pacing.

The sole tick owner advances fixed 50ms simulation steps. Report observed Apply
intervals, but do not interpret accelerated actuator probes as 20Hz runtime.
"""
import argparse,json,time,math
from dataclasses import asdict
from pathlib import Path
import numpy as np,yaml,carla
from simulator.session import CarlaSession
from simulator.m5b_runtime import snapshot_state_measurement
from simulator.m6c_geometry import road_grade
from simulator.m6d_state import signed_from_measurement
from simulator.m6d_motion import HoldEvidence,SignedGuard
from tools.m47_common import physics_dict,freeze_sources,jsonl,dynamics_metrics
from tools.m6d_common import write,asset,provenance
from diagnostics.m6a_oracle_metrics import quantiles


def tasks(protocol,split):
    for pos in protocol[split+'_positions']:
        for seed in protocol['seeds']:
            for candidate in protocol['hold_protocols']:yield dict(split=split,spawn=pos,seed=seed,**candidate)
    for pos in protocol['rollback_diagnostics']['positions']:
        if pos not in protocol[split+'_positions']:continue
        for seed in protocol['seeds']:yield dict(split=split,spawn=pos,seed=seed,name='natural_rollback_protection',warm=False,hold='service',manual1=False,throttle=0.,rollback=True)


def run(task,protocol):
    name=f"{task['split']}_p{task['spawn']}_s{task['seed']}_{task['name']}"
    root=Path('reports/m6d_probe_artifacts')/name;root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',task)
    freeze_sources(root/'source',[__file__,'simulator/m6d_state.py','simulator/m6d_motion.py','simulator/m6d_targets.py','simulator/m5b_runtime.py','simulator/session.py','tools/m47_common.py','configs/m6d/protocol.json'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03',seed=task['seed'])
    with CarlaSession(cfg,load_map=True) as session:
        w=session.world;m=w.get_map();w.set_weather(carla.WeatherParameters.ClearNoon)
        spawn=m.get_spawn_points()[task['spawn']];ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn)
        write(root/'physics.json',physics_dict(ego));assert asset(root/'physics.json')['sha256']==protocol['physics_sha256']
        rows=[];observer=HoldEvidence();guard=SignedGuard();phase='initialize';phase_tick=0;phase_started=None;anchor=None;stable=0
        heading=-math.radians(spawn.rotation.yaw);origin=np.array([spawn.location.x,-spawn.location.y]);route=np.array([np.r_[origin,heading],np.r_[origin+100*np.array([math.cos(heading),math.sin(heading)]),heading]])
        phases=[];collisions=[]
        sensor=session.spawn(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego)
        sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id)))
        for tick in range(1100):
            begin=time.perf_counter();snap=w.get_snapshot()
            if snap.find(ego.id) is None:w.tick();continue
            state,r=snapshot_state_measurement(snap,ego.id,ego.get_control());now=state.timestamp_s
            if phase_started is None:phase_started=now;phases.append(dict(phase=phase,t=now,frame=snap.frame))
            pos=np.array(r['position'][:2])*[1,-1]
            if anchor is None:anchor=pos.copy()
            grade=road_grade(m,r['position']);distance=float((anchor-pos)@np.array([math.cos(heading),math.sin(heading)]))
            s=signed_from_measurement(r,route,distance,grade['grade']);snapshot_end=time.perf_counter()
            transition=None
            if phase=='initialize' and phase_tick>=30:transition='prepare' if task['warm'] else 'hold_initial'
            elif phase=='prepare' and (s.v_forward_mps>=1 or phase_tick>=160):transition='normal_stop'
            elif phase=='normal_stop':
                stable=stable+1 if s.speed_norm_mps<=.02 else 0
                if stable>=5 or phase_tick>=120:transition='hold_warm'
            elif phase in ['hold_initial','hold_warm','hold_final'] and phase_tick>=60:
                transition='release_rollback' if task.get('rollback') else 'restart' if phase=='hold_final' else 'release'
            elif phase in ['release','restart'] and phase_tick>=120:transition='done' if phase=='restart' else 'brake_stop'
            elif phase=='brake_stop':
                stable=stable+1 if s.speed_norm_mps<=.02 else 0
                if stable>=5 or phase_tick>=120:transition='hold_final'
            elif task.get('guarded_release') and phase in ['release','restart'] and guard.latched:transition='protected_stop'
            elif phase=='release_rollback' and (guard.latched or phase_tick>=200):transition='protected_stop' if guard.latched else 'done'
            elif phase=='protected_stop' and phase_tick>=60:transition='done'
            if transition:
                if transition=='done':break
                phase=transition;phase_tick=0;phase_started=now;stable=0;phases.append(dict(phase=phase,t=now,frame=snap.frame));
                if phase.startswith('hold'):anchor=pos.copy();observer=HoldEvidence()
            # Use the new hold anchor on the transition frame itself.
            distance=float((anchor-pos)@np.array([math.cos(heading),math.sin(heading)]))
            s=signed_from_measurement(r,route,distance,grade['grade'])
            u=dict(throttle=0.,brake=0.,steer=0.,hand_brake=False,reverse=False,manual_gear_shift=task['manual1'],gear=1 if task['manual1'] else 0)
            if phase=='initialize':u.update(brake=1.,hand_brake=True)
            elif phase=='prepare':u['throttle']=.20
            elif phase in ['normal_stop','brake_stop']:u['brake']=.01
            elif phase.startswith('hold'):
                u.update(brake=.12 if task['hold']=='service' else 0.,hand_brake=task['hold']=='hand')
            elif phase in ['release','restart']:u['throttle']=task['throttle']
            protection=None
            if (task.get('rollback') and phase in ['release_rollback','protected_stop']) or (task.get('guarded_release') and phase in ['release','restart','protected_stop']):
                protection=guard.check(s,now)
                if protection or phase=='protected_stop':u.update(throttle=0.,brake=1.);protection=protection or 'RollbackDetected'
            mode=observer.update(s,pos,distance,phase.startswith('hold'),phase in ['release','restart','release_rollback'],True,protection)
            controller_end=time.perf_counter();apply=time.perf_counter();ego.apply_control(carla.VehicleControl(**u));apply_end=time.perf_counter();tick_start=time.perf_counter();n=w.tick();tick_end=time.perf_counter()
            rows.append(dict(source='Privileged',frame=snap.frame,next_frame=n,measurement=r,signed=asdict(s),phase=phase,phase_tick=phase_tick,command=u,hold_mode=mode,protection=protection,road=grade,
                reference_position_hs=anchor.tolist(),distance_to_anchor_m=distance,plan_source='NotApplicable_open_loop_actuator_schedule',
                timeline=dict(begin=begin,snapshot_end=snapshot_end,controller_end=controller_end,apply_control=apply,apply_end=apply_end,tick_start=tick_start,tick_end=tick_end)))
            phase_tick+=1
            if collisions:break
        jsonl(root/'trace.jsonl',rows);write(root/'phases.json',phases);write(root/'hold_transitions.json',observer.transitions);write(root/'rollback_events.json',guard.events)
        write(root/'result.json',dict(**provenance('actual_CARLA_fixed_sim_dt_unpaced_actuator_probe'),name=name,task=task,trace=asset(root/'trace.jsonl'),physics=asset(root/'physics.json'),
            ticks=len(rows),collisions=collisions,complete=transition=='done',raw_all_including_initialize=dynamics_metrics([q['measurement'] for q in rows]),
            apply_interval_ms=quantiles(np.diff([q['timeline']['apply_control'] for q in rows])*1000),
            runtime_20Hz='NotApplicable_unpaced_probe',limits='Natural initialization and real snapshots. Diagnostic gear/handbrake only; not a normal controller or runtime success.'))
    return root/'result.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['development','validation']);a=p.parse_args()
    protocol=json.loads(Path('configs/m6d/protocol.json').read_text());plan=list(tasks(protocol,a.split))
    write(f'reports/m6d_probe_{a.split}_plan.json',dict(**provenance('predeclared_probe_tasks'),tasks=plan))
    results=[]
    for i,task in enumerate(plan):
        out=run(task,protocol);results.append(asset(out));print(i+1,len(plan),out,flush=True)
    write(f'reports/m6d_probe_{a.split}_execution.json',dict(**provenance('actual_CARLA_probe_completion'),results=results))
if __name__=='__main__':main()
