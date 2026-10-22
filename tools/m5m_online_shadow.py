"""Actual independent 2Hz six-RGB shadow under a sole dedicated tick owner."""

import time, json, queue, threading, multiprocessing as mp
from pathlib import Path
import numpy as np, yaml
from tools.m5l_common import write, asset, summary
from simulator.m5m_shadow import worker, latest_put


def main():
    import sys

    sys.path.append(str(Path(".venv/lib/python3.10/site-packages").resolve()))
    import carla
    from simulator.session import CarlaSession
    from geometry.reference import intrinsics_from_hfov, camera_mount_to_hs_from_optical
    from geometry.coordinates import carla_pose_to_hs

    ctx = mp.get_context("spawn")
    proc = None
    channel = None
    results = None
    stop = None
    ready = None
    cfg = yaml.safe_load(Path("configs/carla.yaml").read_text())
    cfg.update(port=2100, map="Town01")
    rig = yaml.safe_load(Path("configs/cameras.yaml").read_text())
    packets = {}
    lock = threading.Lock()
    measure = []
    received = []
    sent = 0
    drops = 0
    aborted = False
    startup = []
    setup = []
    try:
        with CarlaSession(cfg, load_map=True) as session:
            world = session.world
            pose = world.get_map().get_spawn_points()[0]
            R = np.array(pose.get_matrix())
            K = []
            T = []

            def callback(name, image):
                wall = time.perf_counter()
                raw = bytes(image.raw_data)
                with lock:
                    packet = packets.setdefault(
                        image.frame,
                        dict(timestamp=image.timestamp, wall=wall, buffers={}),
                    )
                    packet["buffers"][name] = raw
                    packet.setdefault("sensor_timestamps", {})[name] = image.timestamp
                    for frame in list(packets):
                        if frame < image.frame - 3:
                            del packets[frame]

            for name, mount in rig["cameras"].items():
                local = carla.Transform(
                    carla.Location(*mount["xyz_m"]),
                    carla.Rotation(yaw=mount["yaw_deg"], pitch=mount["pitch_deg"]),
                )
                matrix = R @ np.array(local.get_matrix())
                rot = matrix[:3, :3]
                tf = carla.Transform(
                    carla.Location(*matrix[:3, 3]),
                    carla.Rotation(
                        pitch=float(np.degrees(np.arcsin(rot[2, 0]))),
                        yaw=float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                        roll=float(np.degrees(np.arctan2(-rot[2, 1], rot[2, 2]))),
                    ),
                )
                bp = world.get_blueprint_library().find("sensor.camera.rgb")
                for key, value in dict(
                    image_size_x=640,
                    image_size_y=384,
                    fov=100,
                    sensor_tick=0,
                    lens_k=0,
                    lens_kcube=0,
                    motion_blur_intensity=0,
                ).items():
                    bp.set_attribute(key, str(value))
                sensor = session.spawn(bp, tf)
                sensor.listen(lambda im, n=name: callback(n, im))
                K.append(intrinsics_from_hfov(640, 384, 100).tolist())
                T.append(camera_mount_to_hs_from_optical(local.get_matrix()).tolist())
            values = dict(
                intrinsics=(K, "float32"),
                cam_to_ego=(T, "float32"),
                ego_pose=(carla_pose_to_hs(R).tolist(), "float64"),
                ego_state=([0] * 7, "float32"),
                timestamps_s=(1.0, "float64"),
                route_command=(0, "int64"),
                valid_cameras=([True] * 6, "bool"),
            )
            metadata = {k: dict(value=[v], dtype=t) for k, (v, t) in values.items()}
            last_sent = -1
            last_send_wall = -1.0
            phases = [
                (rep, phase)
                for rep, order in enumerate(
                    [
                        ["alone", "loaded_idle", "shadow", "recovery"],
                        ["alone", "shadow", "loaded_idle", "recovery"],
                    ]
                )
                for phase in order
            ]
            for rep, phase in phases:
                if phase in ["alone", "recovery"] and proc is not None:
                    setup_start = time.perf_counter()
                    stop.set()
                    proc.join(10)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(5)
                    setup.append(
                        dict(
                            repeat=rep,
                            phase=phase,
                            worker_shutdown_ms=(time.perf_counter() - setup_start)
                            * 1000,
                        )
                    )
                    proc = None
                if phase in ["loaded_idle", "shadow"] and proc is None and not aborted:
                    channel = ctx.Queue(1)
                    results = ctx.Queue(128)
                    stop = ctx.Event()
                    ready = ctx.Event()
                    proc = ctx.Process(
                        target=worker, args=(channel, results, stop, ready)
                    )
                    proc.start()
                    init_start = time.perf_counter()
                    init_ticks = []
                    while not ready.is_set():
                        t = time.perf_counter()
                        world.tick()
                        init_ticks.append((time.perf_counter() - t) * 1000)
                        if time.perf_counter() - init_start > 90 or not proc.is_alive():
                            stop.set()
                            proc.terminate()
                            raise TimeoutError("Workerinit")
                        time.sleep(max(0, 0.05 - (time.perf_counter() - t)))
                    startup.append(
                        dict(
                            repeat=rep,
                            phase=phase,
                            elapsed_s=time.perf_counter() - init_start,
                            tick_ms=summary(init_ticks),
                            raw_tick_ms=init_ticks,
                        )
                    )
                times = []
                intervals = []
                queue_times = []
                drain_times = []
                prev = None
                for i in range(180):
                    begin = time.perf_counter()
                    fid = world.tick()
                    tick_end = time.perf_counter()
                    packet = None
                    queue_start = time.perf_counter()
                    if (
                        phase == "shadow"
                        and proc is not None
                        and not aborted
                        and begin - last_send_wall >= 0.5
                    ):
                        with lock:
                            complete = [
                                f
                                for f, p in packets.items()
                                if len(p["buffers"]) == 6
                                and max(p["sensor_timestamps"].values())
                                - min(p["sensor_timestamps"].values())
                                < 1e-6
                                and f > last_sent
                            ]
                            if complete:
                                f = max(complete)
                                p = packets[f]
                                packet = dict(
                                    frame=f,
                                    timestamp_s=p["timestamp"],
                                    captured_wall=p["wall"],
                                    buffers=[p["buffers"][n] for n in rig["cameras"]],
                                    metadata=metadata,
                                )
                                last_sent = f
                        if packet:
                            ok = latest_put(channel, packet)
                            sent += int(ok)
                            drops += int(not ok)
                            last_send_wall = begin
                    queue_ms = (time.perf_counter() - queue_start) * 1000
                    drain_start = time.perf_counter()
                    if results is not None:
                        while True:
                            try:
                                received.append(
                                    dict(results.get_nowait(), repeat=rep, phase=phase)
                                )
                            except queue.Empty:
                                break
                    drain_ms = (time.perf_counter() - drain_start) * 1000
                    if i >= 20:
                        times.append((tick_end - begin) * 1000)
                        queue_times.append(queue_ms)
                        drain_times.append(drain_ms)
                        if prev is not None:
                            intervals.append((begin - prev) * 1000)
                    prev = begin
                    if phase in ["shadow", "loaded_idle"] and not aborted:
                        base = next(
                            m["tick_ms"]["p95"]
                            for m in measure
                            if m["repeat"] == rep and m["phase"] == "alone"
                        )
                        p95 = float(np.quantile(times, 0.95)) if times else 0.0
                        if (i >= 60 and p95 > max(base * 1.5, base + 20)) or (
                            tick_end - begin
                        ) * 1000 > 1000:
                            aborted = True
                            stop.set()
                            break
                    time.sleep(max(0, 0.05 - (time.perf_counter() - begin)))
                measure.append(
                    dict(
                        repeat=rep,
                        phase=phase,
                        worker_running=proc is not None and not aborted,
                        queue_enqueue_ms=summary(queue_times),
                        output_drain_ms=summary(drain_times),
                        tick_ms=summary(times),
                        interval_ms=summary(intervals),
                        raw_tick_ms=times,
                        raw_interval_ms=intervals,
                    )
                )
    finally:
        if proc is not None:
            stop.set()
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
        if results is not None:
            while True:
                try:
                    received.append(results.get_nowait())
                except queue.Empty:
                    break
    raw = "runs/m5m_logs/shadow_raw.json"
    write(
        raw,
        dict(
            phases=measure, outputs=received, startup=startup, between_phase_setup=setup
        ),
    )
    write(
        "reports/m5m_shadow_runtime.json",
        dict(
            kind="actual_CARLA_online_sixcamera_independent_process_no_control",
            phases=[
                {k: v for k, v in r.items() if not k.startswith("raw")} for r in measure
            ],
            startup=startup,
            between_phase_setup=setup,
            sent=sent,
            owner_nonblocking_queue_drops=drops,
            outputs=len(received),
            status_counts={
                s: sum(r["status"] == s for r in received)
                for s in {r["status"] for r in received}
            },
            timing={
                key: summary([r[key] for r in received if key in r])
                for key in [
                    "queue_age_ms",
                    "convert_h2d_ms",
                    "inference_ms",
                    "output_ms",
                    "end_to_end_ms",
                ]
            },
            stopped_for_tick_tail=aborted,
            worker_exitcode=None if proc is None else proc.exitcode,
            raw=asset(raw),
            cold=[r for r in received if r.get("cold_start")],
            initialization=[r for r in received if r["status"] == "worker_initialized"],
            steady={
                k: summary(
                    [
                        r[k]
                        for r in received
                        if k in r and not r.get("cold_start", False)
                    ]
                )
                for k in ["inference_ms", "end_to_end_ms", "queue_age_ms"]
            },
            limits="Controlled static trial: process shutdown/setup between phases explicitly timed; no real driving schedulability claim. Queue put duration excludes feeder-thread serialization; total queue age includes sensorbuffer availability, scheduling and IPC. CUDA startup ticks separate from steady ticks. End-to-end startsfirstsensorcallback, not renderingstart. StaticdetachedRGBsensors no drivingroute/controller resourcecontention. Sole harness TickOwner, worker noCARLAclient/no control. 2Hz wallclock throttle, originalsimtimestamp unchanged; agealso usesmonotonicwall. Rawbytes copiedcallback, MP queue serializedasync boundedlatestcomplete. 20warmupticks perphase, finite160ticks, repeatedwithreversedloaded/shadoworder. Doesnotcertify drivingreal-time or purevisualsafety.",
        ),
    )


if __name__ == "__main__":
    main()
