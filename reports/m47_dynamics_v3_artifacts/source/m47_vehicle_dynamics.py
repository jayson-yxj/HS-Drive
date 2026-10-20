"""Actual CARLA open-loop longitudinal system identification at 20 Hz."""
import argparse,json,math
from pathlib import Path
import numpy as np,yaml
from simulator.session import CarlaSession
from simulator.oracle_driver import PurePursuitPID,local_trajectory
from tools.run_oracle_closedloop import find_routes,state_of
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import physics_dict,sample_vehicle,dynamics_metrics,jsonl,freeze_sources


def experiments():
    cases=[]
    for speed in [.5,1.,1.5,2.,4.,5.]:
        cases.append(dict(name=f"coast_v{speed}",v=speed,kind="coast",duration=8.))
        for brake in [.005,.02,.05,.1,.2,.4,1.]:
            if speed in [.5,1.5,4.] or brake in [.02,.1,1.]:
                cases.append(dict(name=f"brake{brake}_v{speed}",v=speed,kind="brake",brake=brake,duration=5.))
    for throttle in [.05,.1,.2,.3,.5]:
        cases.append(dict(name=f"throttle{throttle}",v=0.,kind="throttle",throttle=throttle,duration=6.))
    for rate in [.05,.2,.8]:
        cases.append(dict(name=f"ramp{rate}",v=4.,kind="ramp",rate=rate,duration=7.))
    cases.extend([dict(name="small_throttle",v=2.,kind="small_throttle",duration=8.),
                  dict(name="hold_release",v=1.,kind="hold_release",duration=10.)])
    return cases


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default="reports/m47_vehicle_dynamics.json")
    p.add_argument("--artifacts",default="reports/m47_dynamics_artifacts");p.add_argument("--cases",nargs="*");a=p.parse_args()
    out=Path(a.artifacts)
    if out.exists() or Path(a.output).exists():raise FileExistsError("New paths required")
    out.mkdir();sources=freeze_sources(out/"source",[__file__,"tools/m47_common.py","simulator/session.py"])
    cfg=yaml.safe_load(Path("configs/carla.yaml").read_text());cfg.update(port=2100,map="Town03")
    results=[];physics=None
    for case in experiments():
        if a.cases and case["name"] not in a.cases:continue
        with CarlaSession(cfg,load_map=True) as session:
            world=session.world;world.set_weather(carla.WeatherParameters.ClearNoon)
            _,spawn,path=find_routes(world.get_map(),80)["straight"]
            route=np.array([[w.transform.location.x,-w.transform.location.y,-math.radians(w.transform.rotation.yaw)] for w in path])
            ego=session.spawn(world.get_blueprint_library().find("vehicle.tesla.model3"),spawn)
            ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(30):world.tick()
            pc=physics_dict(ego)
            if physics is None:physics=pc;atomic_json(out/"physics.json",pc)
            if pc!=physics:raise ValueError("Physics changed between experiments")
            ctl=PurePursuitPID();steady=0;warm_rows=[]
            if case["v"]:
                for warm in range(600):
                    st=state_of(world.get_snapshot(),ego);tr,_=local_trajectory(route,st,max_speed=case["v"]);u=ctl.command(tr,st)
                    # M46 PID cannot sustain 0.5m/s without brake stiction.
                    # Low-speed identification uses a measured-speed launch,
                    # not artificial velocity assignment; retain full warmup.
                    throttle=u["throttle"]
                    brake=u["brake"]
                    warm_rows.append(dict(t=st.timestamp_s,v=st.speed_mps,throttle=throttle,brake=brake))
                    ego.apply_control(carla.VehicleControl(throttle=throttle,brake=brake,steer=u["steer"]))
                    world.tick();steady=steady+1 if abs(st.speed_mps-case["v"])<.3 else 0
                    if steady>=10:break
                jsonl(out/(case["name"]+"_warmup.jsonl"),warm_rows)
                if steady<10:raise RuntimeError("Warmup could not stabilize")
            rows=[];start=world.get_snapshot().timestamp.elapsed_seconds
            collisions=[];sensor=session.spawn(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),parent=ego)
            sensor.listen(lambda e:collisions.append(e.frame))
            for tick in range(int(case["duration"]/.05)+1):
                r=sample_vehicle(world,ego);elapsed=r["t"]-start
                st=state_of(world.get_snapshot(),ego);tr,_=local_trajectory(route,st,max_speed=4.);u=ctl.command(tr,st)
                kind=case["kind"];throttle=0.;brake=0.
                if kind=="brake":brake=case["brake"]
                elif kind=="throttle":throttle=case["throttle"]
                elif kind=="ramp":brake=min(.4,case["rate"]*elapsed)
                elif kind=="small_throttle":throttle=.15 if elapsed<2 or elapsed>=4 else .17
                elif kind=="hold_release":
                    brake=.1 if elapsed<3 else .2 if elapsed<5 else 0.
                    throttle=.2 if elapsed>=5 else 0.
                command=dict(throttle=throttle,brake=brake,steer=u["steer"],hand_brake=False)
                r.update(elapsed=elapsed,command=command,planned_speed=None,planned_acceleration=None)
                rows.append(r);ego.apply_control(carla.VehicleControl(**command));world.tick()
            log=out/(case["name"]+".jsonl");jsonl(log,rows)
            result=dict(case=case,initial_speed=rows[0]["long_speed"],final_speed=rows[-1]["long_speed"],
                metrics=dynamics_metrics(rows),collisions=len(collisions),log=str(log),sha256=file_hash(log))
            results.append(result);atomic_json(out/(case["name"]+"_result.json"),result)
            print(case["name"],result["metrics"],flush=True)
    atomic_json(a.output,dict(kind="actual_carla_open_loop",carla="0.9.15",map="Town03",vehicle="vehicle.tesla.model3",dt=.05,
        physics=physics,physics_sha256=file_hash(out/"physics.json"),results=results,sources=sources,
        limitations=["Physics default identical to M46; signed longitudinal velocity is distinct from 3D speed norm.",
        "API acceleration and derivative reported separately; raw values authoritative, smoothing supplemental.",
        "Each experiment begins after measured warmup; no set_target_velocity or altered physics."]))


if __name__=="__main__":main()
