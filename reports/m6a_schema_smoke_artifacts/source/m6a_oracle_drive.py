"""M6-A privileged regression instrumentation; legacy controller and worker unchanged."""

import argparse, json, time, math, threading
from dataclasses import asdict
from pathlib import Path
import numpy as np, yaml
import carla
from simulator.session import CarlaSession
from simulator.m5e_affinity import AffinityPlanner as AsyncPlanner
from simulator.m5a_async import validate_result
import os
from simulator.m48_motion import RouteController
from simulator.m47_motion import continuous_rectangles
from simulator.oracle_driver import footprint
from simulator.motion_oracle import route_projection
from tools.run_oracle_closedloop import obs_of
from simulator.m5b_runtime import snapshot_state_measurement
from tools.m47_common import (
    sample_vehicle,
    physics_dict,
    jsonl,
    freeze_sources,
    dynamics_metrics,
)
from tools.m47_stopping import traffic_lights
from simulator.recorder import atomic_json, file_hash
from datasets.m6a_interfaces import OracleSceneState
from tools.m6a_common import provenance


def stats(x):
    return (
        dict(
            p50=float(np.quantile(x, 0.5)),
            p95=float(np.quantile(x, 0.95)),
            p99=float(np.quantile(x, 0.99)),
            max=float(np.max(x)),
            n=len(x),
        )
        if len(x)
        else None
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m6a_oracle_raw.json")
    p.add_argument("--artifacts", default="reports/m6a_oracle_artifacts")
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=[
            "straight",
            "mild_curve",
            "junction_turn",
            "dynamic_lead",
            "low_speed_stop",
            "worker_crash",
            "emergency",
            "static_obstacle",
            "stalled_worker",
            "invalid_input",
        ],
    )
    p.add_argument("--ticks", type=int, default=0)
    p.add_argument("--hold-after-route", action="store_true")
    p.add_argument("--load-description", default="isolated Oracle; no GPU inference")
    p.add_argument("--route-distance", type=float, default=0)
    p.add_argument('--tick-cpus',default='')
    p.add_argument('--worker-cpus',default='')
    p.add_argument('--lead-motion', choices=['actuator', 'scripted'], default='actuator',
                   help='Scripted lead is an explicit privileged kinematic fixture, not vehicle dynamics validation')
    a = p.parse_args()
    if a.tick_cpus:
        cpus=set(map(int,a.tick_cpus.split(',')))
        for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
    os.environ['M5E_WORKER_CPUS']=a.worker_cpus
    root = Path(a.artifacts)
    if root.exists() or Path(a.output).exists():
        raise FileExistsError("new paths required")
    root.mkdir()
    sources = freeze_sources(
        root / "source",
        [
            __file__,
            "datasets/m6a_interfaces.py",
            "configs/m6a/protocol.json",
            "simulator/m5a_async.py",
            "simulator/m5b_runtime.py",
            "simulator/m48_schedule.py",
            "simulator/m48_motion.py",
            "simulator/m47_motion.py",
            "simulator/motion_oracle.py",
        ],
    )
    cfg = yaml.safe_load(Path("configs/carla.yaml").read_text())
    cfg.update(port=2100, map="Town03")
    results = []
    for scene in a.scenarios:
        route_name = (
            scene
            if scene in ["straight", "mild_curve", "junction_turn"]
            else "straight"
        )
        saved = json.loads(
            Path(
                "reports/m45_oracle_driving_artifacts", route_name + "_route.json"
            ).read_text()
        )
        route = np.array(saved["route_world_hs"])
        route = route[
            np.r_[True, np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) > 1e-5]
        ]
        if scene == "low_speed_stop":
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))]
            route = route[arc <= 9.0]
        with CarlaSession(cfg, load_map=True) as session:
            w = session.world
            cmap = w.get_map()
            w.set_weather(carla.WeatherParameters.ClearNoon)
            if a.route_distance:
                wp = cmap.get_waypoint(
                    cmap.get_spawn_points()[saved["spawn_index"]].location
                )
                points = []
                for _ in range(int(a.route_distance)):
                    t = wp.transform
                    points.append(
                        [t.location.x, -t.location.y, -np.radians(t.rotation.yaw)]
                    )
                    nxt = wp.next(1.0)
                    if not nxt:
                        break
                    wp = min(
                        nxt,
                        key=lambda v: abs(
                            (v.transform.rotation.yaw - t.rotation.yaw + 180) % 360
                            - 180
                        ),
                    )
                route = np.array(points)
                atomic_json(
                    root / (scene + "_extended_route.json"),
                    dict(route=route.tolist(), requested_distance=a.route_distance),
                )
            with traffic_lights(w, True):
                ego = session.spawn(
                    w.get_blueprint_library().find("vehicle.tesla.model3"),
                    cmap.get_spawn_points()[saved["spawn_index"]],
                )
                ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                for _ in range(30):
                    w.tick()
                atomic_json(root / (scene + "_physics.json"), physics_dict(ego))
                obstacles = []
                if scene in ["static_obstacle","dynamic_lead"]:
                    at = route[
                        np.argmin(
                            abs(
                                np.linalg.norm(route[:, :2] - route[0, :2], axis=1) - 20
                            )
                        )
                    ]
                    wp = cmap.get_waypoint(
                        carla.Location(
                            x=float(at[0]), y=float(-at[1]), z=ego.get_location().z
                        )
                    )
                    pose = wp.transform
                    pose.location.z += 0.8
                    target = session.spawn(
                        w.get_blueprint_library().find("vehicle.audi.a2"), pose
                    )
                    target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                    for _ in range(30):
                        w.tick()
                    if scene=='dynamic_lead':
                        target.apply_control(carla.VehicleControl(throttle=.18))
                        if a.lead_motion == 'scripted':
                            target.enable_constant_velocity(carla.Vector3D(1.5, 0, 0))
                    else:
                        target.set_simulate_physics(False)
                    obstacles = [target]
                collisions = []
                lanes = []
                sensor = session.spawn(
                    w.get_blueprint_library().find("sensor.other.collision"),
                    carla.Transform(),
                    parent=ego,
                )
                sensor.listen(lambda e: collisions.append(dict(frame=e.frame, other_id=e.other_actor.id, other_type=e.other_actor.type_id, impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z], source="Privileged")))
                sensor = session.spawn(
                    w.get_blueprint_library().find("sensor.other.lane_invasion"),
                    carla.Transform(),
                    parent=ego,
                )
                sensor.listen(lambda e: lanes.append(dict(frame=e.frame, markings=[str(m.type) for m in e.crossed_lane_markings], source="Privileged")))
                extent = ego.bounding_box.extent
                worker = AsyncPlanner(
                    cmap.to_opendrive(),
                    route,
                    delay=0.5 if scene == "slow_worker" else 0,
                    max_age=0.35,
                )
                rows = []
                controller = RouteController()
                deadline = time.perf_counter()
                wall_start = deadline
                last_submit = -1
                start_sim = w.get_snapshot().timestamp.elapsed_seconds
                stable = 0
                faulted = False
                fault_wall = None
                degradation_wall = None
                try:
                    for tick in range(
                        a.ticks
                        or (
                            1000
                            if scene in ["straight", "mild_curve", "junction_turn", "dynamic_lead", "low_speed_stop"]
                            else (600 if scene == "static_obstacle" else 160)
                        )
                    ):
                        begin = time.perf_counter()
                        if tick == 65 and scene in [
                            "worker_crash",
                            "emergency",
                            "stalled_worker",
                            "invalid_input",
                        ]:
                            fault_wall = begin
                        if scene == "dynamic_lead":
                            if a.lead_motion == 'scripted':
                                if tick == 300:
                                    obstacles[0].disable_constant_velocity()
                                    obstacles[0].apply_control(carla.VehicleControl(brake=.3))
                            else:
                                lead_speed = obstacles[0].get_velocity().length()
                                obstacles[0].apply_control(carla.VehicleControl(
                                    throttle=float(np.clip((1.5-lead_speed)*.25,0,.3)) if tick < 300 else 0.0,
                                    brake=0.0 if tick < 300 else .3))
                        profile = {}
                        stage = time.perf_counter()
                        snap = w.get_snapshot()
                        profile["get_snapshot_ms"] = (
                            time.perf_counter() - stage
                        ) * 1000
                        stage = time.perf_counter()
                        applied = ego.get_control()
                        profile["get_control_ms"] = (time.perf_counter() - stage) * 1000
                        stage = time.perf_counter()
                        state, measurement = snapshot_state_measurement(
                            snap, ego.id, applied
                        )
                        profile["snapshot_state_measurement_ms"] = (
                            time.perf_counter() - stage
                        ) * 1000
                        stage = time.perf_counter()
                        envelope = OracleSceneState(snap.frame, state.timestamp_s, state.timestamp_s + .35,
                                                    coordinate='hs_world_x_y_left_z_up_m')
                        envelope.validate(state.timestamp_s, consumer="oracle_planner")
                        z = measurement["position"][2]
                        if state.timestamp_s - last_submit >= 0.2 - 1e-6:
                            job = dict(
                                scene=scene,
                                frame=snap.frame,
                                state=asdict(state),
                                input_wall=time.perf_counter(),
                                obstacles=[obs_of(snap, o) for o in obstacles],
                                half_length=extent.x,
                                half_width=extent.y,
                                z=z,
                            )
                            if scene == "stalled_worker" and tick >= 65:
                                job["delay"] = 0.5
                            if scene == "invalid_input" and tick >= 65:
                                job["state"].pop("x")
                            profile["snapshot_serialization_ms"] = (
                                time.perf_counter() - stage
                            ) * 1000
                            stage = time.perf_counter()
                            if worker.submit(job):
                                last_submit = state.timestamp_s
                        profile["submit_ms"] = (time.perf_counter() - stage) * 1000
                        if scene == "worker_crash" and tick == 65:
                            threading.Thread(
                                target=worker.worker.terminate, daemon=True
                            ).start()
                            faulted = True
                        stage = time.perf_counter()
                        tr, reason, result = worker.poll(
                            scene, snap.frame, state, time.perf_counter()
                        )
                        profile["ipc_poll_validate_resample_ms"] = (
                            time.perf_counter() - stage
                        ) * 1000
                        if scene == "emergency" and tick >= 65:
                            reason = "emergency_command"
                            faulted = True
                        check_start = time.perf_counter()
                        boundary = False
                        for x, y in footprint(
                            state.x,
                            state.y,
                            state.yaw,
                            extent.x + 0.2,
                            extent.y + 0.15,
                        ):
                            if (
                                cmap.get_waypoint(
                                    carla.Location(x=float(x), y=float(-y), z=z),
                                    project_to_road=False,
                                    lane_type=carla.LaneType.Driving,
                                )
                                is None
                            ):
                                boundary = True
                                reason = reason or "current_road_boundary"
                        profile["road_query_ms"] = (
                            time.perf_counter() - check_start
                        ) * 1000
                        stage = time.perf_counter()
                        # Short independent emergency guard covers obstacle motion since plan snapshot.
                        for obj in [obs_of(snap, o) for o in obstacles]:
                            if continuous_rectangles(
                                [state.x, state.y],
                                obj["xy"],
                                np.array([math.cos(state.yaw), math.sin(state.yaw)])
                                * state.speed_mps,
                                obj.get("velocity", [0, 0]),
                                [
                                    extent.x + 0.2,
                                    extent.y + 0.15,
                                ],
                                [obj["half_length"], obj["half_width"]],
                                state.yaw,
                                obj["yaw"],
                                0.35,
                            ):
                                reason = "current_collision_guard"

                        profile["actor_guard_ms"] = (time.perf_counter() - stage) * 1000
                        stage = time.perf_counter()
                        remaining = (
                            1000
                            if result is None
                            else max(
                                0,
                                result["meta"].get("remaining_m", 1000)
                                - state.speed_mps
                                * (state.timestamp_s - result["state"]["timestamp_s"]),
                            )
                        )
                        if result is not None and reason is None:
                            reason = validate_result(
                                result,
                                scene,
                                snap.frame,
                                state.timestamp_s,
                                time.perf_counter(),
                                worker.max_age,
                            )
                        command = controller.command(
                            tr, state, reason, remaining_m=remaining
                        )
                        profile["controller_ms"] = (time.perf_counter() - stage) * 1000
                        apply_wall = time.perf_counter()
                        ego.apply_control(
                            carla.VehicleControl(
                                throttle=command["throttle"],
                                brake=command["brake"],
                                steer=command["steer"],
                            )
                        )
                        if (
                            fault_wall is not None
                            and degradation_wall is None
                            and reason is not None
                            and command["brake"] > 0
                        ):
                            degradation_wall = apply_wall
                        compute_ms = (time.perf_counter() - begin) * 1000
                        profile["apply_control_ms"] = (
                            time.perf_counter() - apply_wall
                        ) * 1000
                        stage = time.perf_counter()
                        tick_frame = w.tick()
                        profile["world_tick_ms"] = (time.perf_counter() - stage) * 1000
                        finished = time.perf_counter()
                        deadline += 0.05
                        miss = finished > deadline
                        log_start = time.perf_counter()
                        rows.append(
                            dict(
                                source="Privileged",
                                source_envelope=envelope.to_record(),
                                measurement_coordinate='carla_world_x_y_right_z_up_m',
                                oracle_state_coordinate='hs_world_x_y_left_z_up_m',
                                obstacles=[obs_of(snap, o) for o in obstacles],
                                profile=profile,
                                wakeup_lateness_ms=max(0,begin-(deadline-.05))*1000,
                                dynamic_actor_speeds=[float(o.get_velocity().length()) for o in obstacles],
                                frame=snap.frame,
                                tick_return_frame=tick_frame,
                                server_snapshot_platform_timestamp=snap.timestamp.platform_timestamp,
                                sim_time=state.timestamp_s,
                                wall_begin=begin,
                                control_apply_wall=apply_wall,
                                tick_end_wall=finished,
                                deadline_wall=deadline,
                                deadline_missed=miss,
                                compute_ms=compute_ms,
                                cycle_work_ms=(finished - begin) * 1000,
                                measurement=measurement,
                                command=command,
                                reason=reason,
                                boundary=boundary,
                                plan_frame=None if result is None else result["frame"],
                                plan_input_wall=(
                                    None if result is None else result["input_wall"]
                                ),
                                plan_output_wall=(
                                    None if result is None else result["output_wall"]
                                ),
                                plan_sim_age=(
                                    None
                                    if result is None
                                    else state.timestamp_s
                                    - result["state"]["timestamp_s"]
                                ),
                                plan_wall_age=(
                                    None
                                    if result is None
                                    else apply_wall - result["input_wall"]
                                ),
                            )
                        )
                        profile["log_buffer_ms"] = (
                            time.perf_counter() - log_start
                        ) * 1000
                        profile["log_disk_ms"] = 0.0  # disk I/O after control loop
                        if finished < deadline:
                            time.sleep(deadline - finished)
                        elif finished - deadline > 0.05:
                            deadline = (
                                finished  # record miss, do not burst ticks to hide it
                            )
                        arc, arc_all, _, _ = route_projection(route, [state.x, state.y])
                        completion = arc / arc_all[-1]
                        stable = (
                            stable + 1
                            if state.speed_mps < 0.08 and completion > 0.95
                            else 0
                        )
                        if stable >= 20 and not a.hold_after_route:
                            break
                    wall_elapsed = time.perf_counter() - wall_start
                    sim_elapsed = w.get_snapshot().timestamp.elapsed_seconds - start_sim
                finally:
                    worker.close()
                log = root / (scene + ".jsonl")
                jsonl(log, rows)
                plans = root / (scene + "_plans.json")
                atomic_json(plans, [dict(source="Privileged", **v) for v in worker.history])
                intervals = np.diff([r["control_apply_wall"] for r in rows])
                lat = [
                    (r["output_wall"] - r["start_wall"]) * 1000 for r in worker.history
                ]
                reasons = {
                    k: sum(r["reason"] == k for r in rows)
                    for k in set(r["reason"] for r in rows)
                    if k
                }
                route_path = root / (scene + "_route.json")
                atomic_json(route_path, dict(source="Privileged", map="Town03", route_world_hs=route.tolist()))
                result = dict(
                    source="Privileged",
                    lead_motion=a.lead_motion if scene == 'dynamic_lead' else None,
                    route=str(route_path),
                    collision_events=collisions,
                    lane_events=lanes,
                    moving_ticks=sum(
                        r["measurement"]["long_speed"] > 0.2 for r in rows
                    ),
                    measured_travel_m=float(
                        sum(
                            np.linalg.norm(
                                np.array(rows[j]["measurement"]["position"][:2])
                                - rows[j - 1]["measurement"]["position"][:2]
                            )
                            for j in range(1, len(rows))
                        )
                    ),
                    frame_advancement_errors=sum(
                        r["tick_return_frame"] != r["frame"] + 1 for r in rows
                    ),
                    fault_to_brake_s=(
                        None
                        if fault_wall is None or degradation_wall is None
                        else degradation_wall - fault_wall
                    ),
                    normal_control_expired=sum(
                        r["reason"] is None
                        and (
                            r["plan_wall_age"] is None
                            or r["plan_wall_age"] > 0.35 + 1e-6
                            or r["plan_sim_age"] > 0.35 + 1e-6
                        )
                        for r in rows
                    ),
                    minute_windows=[
                        dict(
                            start_s=t,
                            interval_ms=stats(
                                [
                                    v * 1000
                                    for j, v in enumerate(intervals)
                                    if t
                                    <= rows[j + 1]["control_apply_wall"] - wall_start
                                    < t + 60
                                ]
                            ),
                            deadline_miss_rate=float(
                                np.mean(
                                    [
                                        r["deadline_missed"]
                                        for r in rows
                                        if t <= r["wall_begin"] - wall_start < t + 60
                                    ]
                                )
                            ),
                        )
                        for t in range(0, int(wall_elapsed) + 1, 60)
                        if any(t <= r["wall_begin"] - wall_start < t + 60 for r in rows)
                    ],
                    worker_output_latency_ms=stats(
                        [
                            (r["output_wall"] - r["input_wall"]) * 1000
                            for r in worker.history
                        ]
                    ),
                    load_description=a.load_description,
                    stage_profile_ms={
                        k: stats([r["profile"].get(k, 0.0) for r in rows])
                        for k in set().union(*(r["profile"].keys() for r in rows))
                    },
                    scene=scene,
                    route_completion=completion,
                    route_passed=stable >= 20,
                    collisions=len(collisions),
                    lane_invasions=len(lanes),
                    road_boundary_frames=sum(r["boundary"] for r in rows),
                    ticks=len(rows),
                    sim_hz=len(rows) / sim_elapsed,
                    wall_hz=len(rows) / wall_elapsed,
                    planner_ms=stats(lat),
                    controller_compute_ms=stats([r["compute_ms"] for r in rows]),
                    control_interval_ms=stats(intervals * 1000),
                    deadline_miss_rate=float(
                        np.mean([r["deadline_missed"] for r in rows])
                    ),
                    interval_over_55ms_rate=float(np.mean(intervals > 0.055)),
                    plan_sim_age_s=stats(
                        [
                            r["plan_sim_age"]
                            for r in rows
                            if r["plan_sim_age"] is not None
                        ]
                    ),
                    plan_wall_age_s=stats(
                        [
                            r["plan_wall_age"]
                            for r in rows
                            if r["plan_wall_age"] is not None
                        ]
                    ),
                    degradation_ticks=reasons,
                    rejected=worker.rejected,
                    worker_history=str(plans),
                    log=str(log),
                    log_sha256=file_hash(log),
                    actual_dynamics=dynamics_metrics([r["measurement"] for r in rows]),
                )
                results.append(result)
                atomic_json(root / (scene + "_result.json"), result)
                print(scene, result, flush=True)
    atomic_json(
        a.output,
        dict(
            kind="actual_CARLA_wall_paced_async_process",
            source="Privileged",
            provenance=provenance("actual_CARLA_closed_loop"),
            results=results,
            sources=sources,
            worker_world_tick_calls=0,
            full_safety_horizon_s=6.0,
            control_period_s=0.05,
            planner_period_s=0.2,
            max_plan_age_s=0.35,
            vision_inputs=False,
            load_description=a.load_description,
            limitations=[
                "Privileged oracle route/state, not vision closed loop.",
                "Road checks use corners, not exact swept road mesh.",
                "No unsafe fast road certificates.",
                "Route trials alone do not validate dynamic collision generalization.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
