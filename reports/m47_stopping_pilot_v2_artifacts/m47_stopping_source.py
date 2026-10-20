"""Matched M4.6/M4.7 CARLA braking and trajectory tests; immutable outputs."""

import argparse, json, math, time, sys
import shutil
from contextlib import contextmanager
from pathlib import Path
import numpy as np, yaml
from simulator.session import CarlaSession
from simulator.recorder import atomic_json, file_hash
from simulator import oracle_driver as old
from simulator import motion_oracle as new
from simulator import m47_motion
from tools.m47_common import physics_dict,sample_vehicle
from tools.run_oracle_closedloop import find_routes, state_of, obs_of, CARLA_AGENTS
from simulator.motion_oracle import cross_track

SCENARIOS = {
    "static_8_v2": dict(gap=8, v=2),
    "static_15_v4": dict(gap=15, v=4),
    "static_30_v5": dict(gap=30, v=5),
    "static_45_v4": dict(gap=45, v=4),
    "lead_braking": dict(gap=12, v=4, lead=True),
    "multi_actor": dict(gap=25, v=4, multi=True),
    "straight": dict(v=0),
    "mild_curve": dict(v=0),
    "junction_turn": dict(v=0),
    "emergency": dict(v=4, fault="emergency_command"),
    "timeout": dict(v=4, fault="trajectory_timeout"),
    "invalid": dict(v=4, fault="invalid_trajectory"),
    "missing": dict(v=4, fault="no_trajectory"),
}


@contextmanager
def traffic_lights(world, green=False):
    """Optional fixed-green route regression; restores actors on exit.

    This diagnoses steering/road geometry only, not traffic-rule compliance.
    """
    import carla

    saved = []
    try:
        if green:
            for light in world.get_actors().filter("traffic.traffic_light"):
                saved.append((light, light.get_state(), light.is_frozen()))
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
        yield
    finally:
        for light, state, frozen in saved:
            if light.is_alive:
                light.set_state(state)
                light.freeze(frozen)


def main():
    import carla

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m47_stopping_validation.json")
    p.add_argument("--artifacts", default="reports/m47_stopping_artifacts")
    p.add_argument("--scenarios", nargs="*")
    p.add_argument("--versions", nargs="*", default=["m46", "m47"])
    p.add_argument("--ackermann", action="store_true")
    p.add_argument("--replay-routes")
    p.add_argument("--green-lights", action="store_true")
    a = p.parse_args()
    if Path(a.output).exists() or Path(a.artifacts).exists():
        raise FileExistsError("New output paths required")
    root = Path(a.artifacts)
    root.mkdir()
    # Snapshot before execution; reports must describe the code actually loaded.
    source_hashes = {}
    for filename in (
        "simulator/m47_motion.py",
        "tools/m47_common.py",
        "reports/m47_dynamics_calibration.json",
        "simulator/motion_oracle.py",
        "simulator/oracle_driver.py",
        __file__,
    ):
        source = Path(filename)
        target = root / (source.stem + "_source.py")
        shutil.copy2(source, target)
        source_hashes[str(source)] = file_hash(source)
    cfg = yaml.safe_load(Path("configs/carla.yaml").read_text())
    cfg.update(port=2100, map="Town03")
    client = carla.Client("127.0.0.1", 2100)
    client.set_timeout(60)
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != "Town03":
        world = client.load_world("Town03")
    cmap = world.get_map()
    routes = find_routes(cmap, 80)
    sys.path.insert(0, CARLA_AGENTS)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    grp = GlobalRoutePlanner(cmap, 1.0)
    results = []
    for name in a.scenarios or SCENARIOS:
        setting = SCENARIOS[name]
        key = name if name in routes else "straight"
        spawnid, spawn, path = routes[key]
        trace = grp.trace_route(path[0].transform.location, path[-1].transform.location)
        route = np.array(
            [
                [
                    w.transform.location.x,
                    -w.transform.location.y,
                    -math.radians(w.transform.rotation.yaw),
                ]
                for w, _ in trace
            ]
        )
        if a.replay_routes:
            saved = json.loads(
                (Path(a.replay_routes) / (key + "_route.json")).read_text()
            )
            if saved["map"] != cmap.name:
                raise ValueError("Replay map mismatch")
            spawnid = saved["spawn_index"]
            spawn = cmap.get_spawn_points()[spawnid]
            route = np.asarray(saved["route_world_hs"], dtype=float)
        route = route[
            np.r_[True, np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) > 1e-5]
        ]
        for version in a.versions:
            tag = version + "_" + name
            rows = []
            collisions = []
            lanes = []
            reason_counts = {}
            stable = 0
            failure = None
            cache = {}
            with (
                CarlaSession(cfg, load_map=False) as session,
                traffic_lights(world, a.green_lights),
            ):
                world = session.world
                world.set_weather(carla.WeatherParameters.ClearNoon)
                bp = world.get_blueprint_library().find("vehicle.tesla.model3")
                ego = session.spawn(bp, spawn)
                ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                for _ in range(30):
                    world.tick()
                # Identical physics warmup for both versions. One-shot velocity
                # assignment caused tire transients in pilot; keep that evidence,
                # but compare measured stabilized speeds in final experiments.
                warm = old.PurePursuitPID()
                steady = 0
                warm_rows = []
                if setting["v"] > 0:
                    ego.apply_control(carla.VehicleControl())
                    for warm_tick in range(400):
                        ws = state_of(world.get_snapshot(), ego)
                        wt, _ = old.local_trajectory(route, ws, max_speed=setting["v"])
                        wc = warm.command(wt, ws)
                        ego.apply_control(
                            carla.VehicleControl(
                                steer=wc["steer"],
                                throttle=wc["throttle"],
                                brake=wc["brake"],
                            )
                        )
                        warm_rows.append(dict(v=ws.speed_mps, t=ws.timestamp_s))
                        steady = (
                            steady + 1 if abs(ws.speed_mps - setting["v"]) < 0.25 else 0
                        )
                        if steady >= 12:
                            break
                        world.tick()
                    if steady < 12:
                        atomic_json(root / (tag + "_failed_warmup.json"), warm_rows)
                        raise RuntimeError("Initial speed failed to stabilize")
                targets = []
                lead = None
                if "gap" in setting:
                    wp = cmap.get_waypoint(ego.get_location())
                    tf = wp.next(setting["gap"] + 2 * ego.bounding_box.extent.x)[
                        0
                    ].transform
                    tf.location.z += 0.5
                    lead = session.spawn(bp, tf)
                    lead.apply_control(
                        carla.VehicleControl(
                            brake=1, hand_brake=not setting.get("lead", False)
                        )
                    )
                    targets.append(lead)
                    if not setting.get("lead"):
                        lead.set_simulate_physics(False)
                        tf.location.z -= 0.45
                        lead.set_transform(tf)
                    if setting.get("multi"):
                        for along, side in [(12, 4.5), (38, 0)]:
                            tf = (
                                cmap.get_waypoint(ego.get_location())
                                .next(along)[0]
                                .transform
                            )
                            tf.location.z += 0.5
                            right = tf.get_right_vector()
                            tf.location += carla.Location(
                                x=right.x * side, y=right.y * side
                            )
                            actor = session.spawn(bp, tf)
                            actor.apply_control(
                                carla.VehicleControl(brake=1, hand_brake=True)
                            )
                            targets.append(actor)
                            actor.set_simulate_physics(False)
                            tf.location.z -= 0.45
                            actor.set_transform(tf)
                collision = session.spawn(
                    world.get_blueprint_library().find("sensor.other.collision"),
                    carla.Transform(),
                    parent=ego,
                )
                collision.listen(
                    lambda e: collisions.append(
                        dict(frame=e.frame, other=e.other_actor.type_id)
                    )
                )
                lane = session.spawn(
                    world.get_blueprint_library().find("sensor.other.lane_invasion"),
                    carla.Transform(),
                    parent=ego,
                )
                lane.listen(lambda e: lanes.append(e.frame))
                if setting.get("lead"):
                    lead.apply_control(carla.VehicleControl())
                    lead.enable_constant_velocity(carla.Vector3D(2.5, 0, 0))
                world.tick()
                planner = new.MotionPlanner()
                controller = (
                    new.MotionController() if version in ("m46", "m47") else old.PurePursuitPID()
                )
                if version in ("m46", "m47"):
                    controller.effort = float(
                        ego.get_control().throttle - ego.get_control().brake
                    )
                else:
                    controller.last_throttle = float(ego.get_control().throttle)
                leadcontroller = old.PurePursuitPID()
                if version=="m47":
                    controller=m47_motion.CalibratedController()
                    controller.effort=float(ego.get_control().throttle-ego.get_control().brake)
                physics=physics_dict(ego)
                atomic_json(root/(tag+"_physics.json"),physics)
                start_time = world.get_snapshot().timestamp.elapsed_seconds
                halfl = ego.bounding_box.extent.x
                halfw = ego.bounding_box.extent.y
                for tick in range(850):
                    snap = world.get_snapshot()
                    state = state_of(snap, ego)
                    obstacles = []
                    for target in targets:
                        o = obs_of(snap, target)
                        o["actor_id"] = target.id
                        obstacles.append(o)
                    tick_start=time.perf_counter()
                    road_profile={"api_ms":0.,"calls":0}
                    ego_z=snap.find(ego.id).get_transform().location.z
                    gap = new.clearance(route, state, obstacles, halfl, halfw)
                    start = time.perf_counter()
                    if version in ("m46", "m47"):
                        tr, meta = planner.plan(
                            route,
                            state,
                            obstacles,
                            max_speed=5 if setting["v"] == 5 else 4,
                            half_length=halfl,
                            half_width=halfw,
                        )
                        index = meta["route_index"]
                    else:
                        tr, index = old.local_trajectory(
                            route,
                            state,
                            max_speed=5 if setting["v"] == 5 else 4,
                            stop_distance=gap["gap_m"] if gap else None,
                        )
                        meta = {}
                    planner_ms=(time.perf_counter()-start)*1000
                    fault = setting.get("fault") if tick >= 35 else None
                    if fault == "trajectory_timeout":
                        tr.timestamp_s -= 1
                    if fault == "invalid_trajectory":
                        tr.points[0, 0] = np.nan
                    if fault == "no_trajectory":
                        tr = None

                    def road_ok(x, y):
                        cell = (round(x, 2), round(y, 2))
                        if cell not in cache:
                            rt=time.perf_counter()
                            cache[cell] = (
                                cmap.get_waypoint(
                                    carla.Location(
                                        x=float(x), y=float(-y), z=ego_z
                                    ),
                                    project_to_road=False,
                                    lane_type=carla.LaneType.Driving,
                                )
                                is not None
                            )
                            road_profile["api_ms"]+=(time.perf_counter()-rt)*1000
                            road_profile["calls"]+=1
                        return cache[cell]

                    safety = m47_motion.safety_check if version=="m47" else new.safety_check
                    safety_start=time.perf_counter()
                    reason = (
                        "emergency_command"
                        if fault == "emergency_command"
                        else safety(
                            tr,
                            state,
                            obstacles,
                            half_length=halfl,
                            half_width=halfw,
                            road_predicate=road_ok,
                        )
                    )
                    safety_ms=(time.perf_counter()-safety_start)*1000
                    if ego.is_at_traffic_light() and str(
                        ego.get_traffic_light_state()
                    ) in ("Red", "Yellow"):
                        reason = reason or "traffic_signal_stop"
                    command = controller.command(
                        tr,
                        state,
                        reason,
                        **(
                            {"remaining_m": meta["remaining_m"]}
                            if version in ("m46", "m47")
                            else {}
                        ),
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    if a.ackermann and version in ("m46", "m47") and reason is None:
                        target = float(tr.points[min(4, len(tr.points) - 1), 3])
                        ego.apply_ackermann_control(
                            carla.VehicleAckermannControl(
                                steer=command["steer"] * 0.6,
                                steer_speed=0.9,
                                speed=target,
                                acceleration=1.8,
                                jerk=2.5,
                            )
                        )
                    else:
                        ego.apply_control(
                            carla.VehicleControl(
                                steer=command["steer"],
                                throttle=command["throttle"],
                                brake=command["brake"],
                            )
                        )
                    if setting.get("lead"):
                        ls = state_of(snap, lead)
                        if state.timestamp_s - start_time >= 4:
                            lead.disable_constant_velocity()
                            lead.apply_control(carla.VehicleControl(brake=0.35))
                        else:
                            lt, _ = old.local_trajectory(route, ls, max_speed=2.5)
                            lc = leadcontroller.command(lt, ls)
                            lead.apply_control(
                                carla.VehicleControl(
                                    steer=lc["steer"],
                                    throttle=lc["throttle"],
                                    brake=lc["brake"],
                                )
                            )
                    clean = (
                        None
                        if tr is None
                        else [
                            [float(v) if np.isfinite(v) else None for v in row]
                            for row in tr.points
                        ]
                    )
                    consistency = (
                        None if tr is None else new.validate_kinematics(tr, state)
                    )
                    rows.append(
                        dict(measurement=sample_vehicle(world,ego),planner_ms=planner_ms,safety_ms=safety_ms,road_profile=road_profile,
                             hold_entry=getattr(controller,"hold_entry",None),
                            frame=snap.frame,
                            t=state.timestamp_s - start_time,
                            x=state.x,
                            y=state.y,
                            yaw=state.yaw,
                            v=state.speed_mps,
                            gap=None if gap is None else gap["gap_m"],
                            obstacles=obstacles,
                            trajectory=clean,
                            trajectory_dt=None if tr is None else tr.dt_s,
                            kinematic_violation=consistency,
                            command=command,
                            actual_control=dict(
                                throttle=ego.get_control().throttle,
                                brake=ego.get_control().brake,
                                steer=ego.get_control().steer,
                            ),
                            safety_reason=reason,
                            latency_ms=elapsed_ms,
                            completion=index / (len(route) - 1),
                            plan_remaining=meta.get("remaining_m"),
                        )
                    )
                    stopped = state.speed_mps < 0.08
                    if "fault" in setting:
                        met = stopped and tick > 40
                    elif "gap" in setting:
                        met = (
                            stopped
                            and rows[-1]["gap"] is not None
                            and rows[-1]["gap"] < 20
                            and tick > 100
                        )
                    else:
                        met = (
                            stopped
                            and np.linalg.norm(route[-1, :2] - [state.x, state.y]) < 2.5
                        )
                    if version=="m47" and "gap" in setting and not setting.get("fault"):
                        met=met and controller.mode=="HOLD"
                    stable = stable + 1 if met else 0
                    if collisions:
                        failure = "collision"
                        break
                    if stable >= 30:
                        break
                    if tick == 849:
                        failure = "max_ticks"
                    world.tick()
            time.sleep(0.1)
            if (
                "gap" in setting
                and rows[-1]["gap"] is not None
                and rows[-1]["gap"] < 1.0
            ):
                failure = failure or "unsafe_stopping_clearance"
            file = root / (tag + ".jsonl")
            with file.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            v = np.array([r["v"] for r in rows])
            acc = np.diff(v) / 0.05
            jerk = np.diff(acc) / 0.05
            smooth = np.convolve(v, np.ones(5) / 5, mode="valid")
            sa = np.diff(smooth) / 0.05
            sj = np.diff(sa) / 0.05
            lat = cross_track(np.array([[r["x"], r["y"]] for r in rows]), route)
            event = sum(
                1
                for i, r in enumerate(rows)
                if r["safety_reason"]
                and (i == 0 or rows[i - 1]["safety_reason"] != r["safety_reason"])
            )
            result = dict(
                version=version,
                scenario=name,
                settings=setting,
                replay_routes=a.replay_routes,
                green_lights=a.green_lights,
                ackermann=a.ackermann and version in ("m46", "m47"),
                scope="Same scene/route/state setup; entire planner-controller stack differs, not single-variable ablation",
                status="Passed" if failure is None and not collisions else "Failed",
                failure=failure,
                ticks=len(rows),
                initial_actual_speed_mps=rows[0]["v"],
                initial_actual_gap_m=rows[0]["gap"],
                collision_count=len(collisions),
                lane_invasions=len(lanes),
                final_gap_m=rows[-1]["gap"],
                min_gap_m=min(
                    (r["gap"] for r in rows if r["gap"] is not None), default=None
                ),
                final_speed_mps=float(v[-1]),
                max_speed_mps=float(v.max()),
                route_completion=max(r["completion"] for r in rows),
                max_decel_raw_mps2=float(max(0, -acc.min())),
                jerk_abs_p95_raw_mps3=float(np.quantile(abs(jerk), 0.95)),
                jerk_abs_max_raw_mps3=float(abs(jerk).max()),
                max_decel_250ms_mps2=float(max(0, -sa.min())),
                jerk_abs_p95_250ms_mps3=float(np.quantile(abs(sj), 0.95)),
                lateral_mean_m=float(lat.mean()),
                lateral_max_m=float(lat.max()),
                safety_forced_braking_events=event,
                safety_forced_braking_ticks=sum(reason_counts.values()),
                safety_reasons=reason_counts,
                kinematic_violation_ticks=sum(
                    r["kinematic_violation"] is not None for r in rows
                ),
                latency_p95_ms=float(
                    np.quantile([r["latency_ms"] for r in rows], 0.95)
                ),
                log=str(file),
                log_sha256=file_hash(file),
            )
            results.append(result)
            atomic_json(root / (tag + "_result.json"), result)
            atomic_json(
                root / (tag + "_route.json"),
                dict(route=route.tolist(), spawn_index=spawnid),
            )
            print(tag, json.dumps(result), flush=True)
    atomic_json(
        a.output,
        dict(
            schema_version="hsd-m47-closedloop-v1",
            results=results,
            policy=vars(new.MotionPolicy()),
            base_source_sha256=source_hashes["simulator/oracle_driver.py"],
            new_source_sha256=source_hashes["simulator/motion_oracle.py"],
            source_hashes_at_start=source_hashes,
            source_unchanged_during_run=all(
                file_hash(p) == h for p, h in source_hashes.items()
            ),
            notes=[
                "Simulation step 20Hz; sensor/network latency excluded",
                "Footprints sampled <=0.1m and <=0.05s in M46, not exact continuous collision detection",
                "250ms moving-average kinematics separately labeled; raw values retained",
                "Ego initial speed stabilized by identical old controller before actor placement; lead scripted 2.5m/s for 4s then physics braking",
            ],
        ),
    )


if __name__ == "__main__":
    main()
