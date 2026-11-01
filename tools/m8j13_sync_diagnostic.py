"""Bounded native RGB reproduction. No model, PID, watchdog or actuation loop.

Run through this supervisor, never invoke worker manually. Every start is
consumed before Popen. No load/reload, retry or hidden initialization loop.
World::ApplySettings' <=30 internal Tick calls are conservatively charged.
"""
import argparse
import ast
from collections import Counter, defaultdict, deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from simulator.m8j13_sensor_audit import RawInbox, check_group
from tools.m8j13_common import config, write, atomic, append, digest, provenance

PLAN = Path('configs/m8j13_sync_diagnostic.json')
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def settings_record(world):
    s = world.get_settings()
    return {k: getattr(s, k) for k in ('synchronous_mode', 'fixed_delta_seconds',
            'no_rendering_mode', 'substepping', 'max_substep_delta_time', 'max_substeps')}


def snapshot_record(snapshot, ego=None):
    row = dict(frame=snapshot.frame, elapsed_seconds=snapshot.timestamp.elapsed_seconds,
               delta_seconds=snapshot.timestamp.delta_seconds, ego=None)
    a = snapshot.find(ego.id) if ego is not None else None
    if a is not None:
        v = a.get_velocity()
        row['ego'] = dict(actor_id=ego.id, transform=a.get_transform().get_matrix(),
                          velocity=[v.x, v.y, v.z])
    return row


def native_camera_spec(source):
    tree = ast.parse(Path(source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDriveAgent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sensors')
    scope = {'resize_scale': .44, 'IS_BENCH2DRIVE': False}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return {s['id']: s for s in scope['sensors'](None) if s['type'] == 'sensor.camera.rgb'}


def worker(root):
    import carla
    c = config(); plan = json.loads(PLAN.read_text()); request = json.loads((root/'request.json').read_text())
    spec = plan[request['stage']]; names = CAMERAS if spec['cameras'] == 'all' else spec['cameras']
    client = carla.Client('127.0.0.1', c['port']); client.set_timeout(5.)
    world = client.get_world(); original = world.get_settings(); actors = []; sensors = []; inbox = None
    result = dict(status='Partial', model_forwards=0, model_controls=0, training=0,
                  tick_owner_pid=os.getpid(), traffic_manager_created=False,
                  world_loads=0, world_reloads=0, settings_internal_tick_charge=30)
    snapshots = {}; records = []; ring = deque(maxlen=84); evidence = {}; first_bad = None
    stats = Counter(); candidates = defaultdict(list); hs_events = []; hs_outputs = []; hs = None
    explicit_ticks = 0; window = []; actor_map = {}; identity = None
    def event(kind, **data):
        append(root/'lifecycle.jsonl', dict(event=kind, monotonic_s=time.monotonic(), **data))
    def drain():
        nonlocal first_bad
        for stamp, image in inbox.drain():
            r = stamp.record(); r['record_id'] = len(records)
            r['drained_after_frame'] = world.get_snapshot().frame
            r['object_fields_unchanged'] = (image.frame == stamp.frame and image.timestamp == stamp.timestamp_s
                and tuple(tuple(float(x) for x in row) for row in image.transform.get_matrix()) == stamp.transform)
            records.append(r); candidates[r['frame']].append(r); ring.append((r, image))
            snap = snapshots.get(r['frame'])
            mismatch = snap is not None and abs(r['timestamp_s']-snap['elapsed_seconds']) > plan['tolerance_s']
            duplicate = sum(x['camera'] == r['camera'] for x in candidates[r['frame']]) > 1
            if mismatch: stats['raw_snapshot_mismatches'] += 1
            if duplicate: stats['duplicates'] += 1
            if not r['object_fields_unchanged']: stats['object_field_mutations'] += 1
            if r['drained_after_frame'] > r['frame']: stats['arrived_or_drained_after_sample_frame'] += 1
            if first_bad is None and (mismatch or duplicate or not r['object_fields_unchanged']):
                first_bad = r['frame']
            if first_bad is not None:
                for old, obj in ring:
                    if abs(old['frame']-first_bad) <= 2:
                        evidence[old['record_id']] = (old, obj)
            if hs is not None:
                hs.callback(r['camera'])(image)
        if hs is not None:
            for bundle in hs.drain():
                hs_outputs.append(dict(frame=bundle.frame_id, timestamp_s=bundle.timestamp_s,
                    cameras={n: dict(frame=s.frame_id, timestamp_s=s.timestamp_s) for n, s in bundle.sensors.items()}))
    try:
        assert client.get_client_version() == client.get_server_version() == '0.9.15'
        assert not original.synchronous_mode
        event('initial_world', episode=world.id, settings=settings_record(world), snapshot=snapshot_record(world.get_snapshot()))
        if world.get_map().name.split('/')[-1] != c['map']:
            # Initialization only, while asynchronous: LoadEpisode's conditional
            # SendTickCue is not entered. No retry/reload after measurements.
            event('map_load_requested', map=c['map'], settings=settings_record(world))
            client.set_timeout(45.)
            world = client.load_world(c['map']); result['world_loads'] += 1
            client.set_timeout(5.)
            original = world.get_settings()
            event('map_loaded', episode=world.id, settings=settings_record(world), snapshot=snapshot_record(world.get_snapshot()))
        assert world.get_map().name.split('/')[-1] == c['map'], 'UnexpectedMap'
        assert not original.synchronous_mode and original.fixed_delta_seconds is None
        settings = world.get_settings(); settings.synchronous_mode = True
        settings.fixed_delta_seconds = c['physics_dt_s']; settings.no_rendering_mode = False
        settings.substepping = True; settings.max_substep_delta_time = .01; settings.max_substeps = 10
        before = world.get_snapshot().frame
        world.apply_settings(settings)
        after = world.get_snapshot().frame
        result['apply_settings_observed_frame_delta'] = after-before
        event('settings_applied', settings=settings_record(world), snapshot=snapshot_record(world.get_snapshot()),
              conservative_internal_tick_charge=30, observed_frame_delta=after-before)
        assert settings_record(world) == dict(synchronous_mode=True, fixed_delta_seconds=.05,
            no_rendering_mode=False, substepping=True, max_substep_delta_time=.01, max_substeps=10)
        scene = c['scenes'][0]; route = json.loads(Path(scene['route_file']).read_text())
        xyz = route['route_world_carla'][scene['start_index']]
        yaw = -route['route_world_hs'][scene['start_index']][2]*180/3.141592653589793
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = world.get_blueprint_library().find(c['vehicle'])
        ego = world.spawn_actor(bp, carla.Transform(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]+.5), carla.Rotation(yaw=yaw)))
        actors.append(ego); ego.set_simulate_physics(False)
        identity = (root.name, world.id, root.name+':rgb:1')
        inbox = RawInbox(*identity)
        source = Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py'
        cameras = native_camera_spec(source)
        for name in names:
            s = cameras[name]; bp = world.get_blueprint_library().find('sensor.camera.rgb')
            for key, value in [('image_size_x',704), ('image_size_y',396), ('fov',s['fov']), ('sensor_tick',0.)]:
                bp.set_attribute(key, str(value))
            tf = carla.Transform(carla.Location(x=s['x'],y=s['y'],z=s['z']), carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll']))
            cam = world.spawn_actor(bp, tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            actors.append(cam); sensors.append(cam); actor_map[name] = cam.id
            event('sensor_created', identity=identity, camera=name, actor_id=cam.id, attributes=dict(cam.attributes), relative_transform=tf.get_matrix())
            cam.listen(inbox.callback(name, cam.id))
        if request['stage'] != 'D1':
            from simulator.synchronizer import FrameSynchronizer
            hs = FrameSynchronizer(names, cadence_ticks=1, timeout_ticks=5, max_frames=8, on_event=hs_events.append)
        write(root/'runtime_identity.json', dict(run=root.name, episode=world.id, generation=identity[2], actors=actor_map,
            client=client.get_client_version(), server=client.get_server_version(), python=sys.executable,
            python_version=sys.version, carla_path=carla.__file__, extension_path=carla.libcarla.__file__,
            extension_sha256=digest(carla.libcarla.__file__), settings=settings_record(world),
            map=world.get_map().name, sensor_tick=0., capture_wh=[704,396], camera_order=list(names)))
        last = world.get_snapshot().frame; next_cycle = time.monotonic()
        total = spec['warmup_ticks']+spec['measurement_ticks']+spec['drain_ticks']
        for index in range(total):
            phase = 'warmup' if index < spec['warmup_ticks'] else 'measurement' if index < total-spec['drain_ticks'] else 'drain'
            # Fixed wall pacing in baseline; barrier is an explicit diagnostic
            # condition, never a new freshness threshold or a 20 Hz claim.
            if request['mode'] == 'paced':
                time.sleep(max(0., next_cycle-time.monotonic()))
                next_cycle = time.monotonic()+.05
            start = time.monotonic()
            append(root/'tick_requests.jsonl', dict(index=index, phase=phase, monotonic_s=start))
            explicit_ticks += 1
            frame = world.tick(5.); returned = time.monotonic(); snap = world.get_snapshot()
            if snap.frame != frame or frame != last+1: raise RuntimeError('CompetingTickOwnerOrFrameGap')
            last = frame; row = snapshot_record(snap, ego)
            row.update(phase=phase, tick_start_monotonic_s=start, tick_return_monotonic_s=returned,
                       run=root.name, episode=world.id)
            snapshots[frame] = row
            append(root/'snapshots.jsonl', row)
            if phase == 'measurement': window.append(frame)
            if hs is not None:
                if hs.latest == -1: hs.begin_interval(frame, explicit_cadence=True)
                hs.add_snapshot(frame, row['elapsed_seconds'], row['ego'], [])
            drain()
            if request['mode'] == 'barrier' and index >= 3:
                deadline = time.monotonic()+plan['barrier_wait_wall_s']
                while time.monotonic() < deadline and set(x['camera'] for x in candidates[frame]) != set(names):
                    time.sleep(.001); drain()
                check = check_group(candidates[frame], row, identity, actor_map)
                if not check['valid']:
                    raise RuntimeError('BarrierInvalidGroup:'+','.join(check['errors']))
            if inbox.stats['queue_overflow']: raise RuntimeError('RawQueueOverflow')
        # Bounded final receive-only drain: never tick to hunt for a good window.
        deadline = time.monotonic()+2.
        while time.monotonic() < deadline:
            drain(); time.sleep(.005)
        result['status'] = 'Measured'
    except BaseException as exc:
        result['error'] = repr(exc); result['traceback'] = traceback.format_exc()
    finally:
        if inbox is not None:
            drain(); inbox.close(); event('generation_invalidated', identity=identity, stats=dict(inbox.stats))
        for sensor in sensors:
            try: sensor.stop(); event('listener_stopped', actor_id=sensor.id)
            except Exception as exc: event('cleanup_error', error=repr(exc))
        for actor in reversed(actors):
            try: event('actor_destroyed', actor_id=actor.id, result=actor.destroy())
            except Exception as exc: event('cleanup_error', error=repr(exc))
        # Original has fixed_delta_seconds=None: ApplySettings does not tick.
        world.apply_settings(original)
        fresh = world.wait_for_tick(5.)
        result['cleanup'] = dict(settings=settings_record(world), remaining_owned=[a.id for a in actors if fresh.find(a.id) is not None],
                                 traffic_manager_state='NotCreated')
        event('restored', **result['cleanup'])
        write(root/'raw_images_metadata.json', records)
        groups = [check_group(candidates[f], snapshots[f], identity, actor_map) for f in window]
        write(root/'groups.json', groups); write(root/'hs_groups.json', hs_outputs); write(root/'hs_events.json', hs_events)
        evidence_index = []
        for record_id, (r, image) in sorted(evidence.items()):
            data = bytes(image.raw_data); path = root/'anomaly_images'/('%06d.bgra'%record_id); path.parent.mkdir(exist_ok=True)
            path.write_bytes(data); evidence_index.append(dict(record_id=record_id, frame=r['frame'], camera=r['camera'], file=path.name,
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), width=image.width, height=image.height))
        write(root/'anomaly_images_index.json', evidence_index)
        result.update(explicit_tick_requests=explicit_ticks, charged_active_ticks=explicit_ticks+30,
            measurement_world_ticks=len(window), configured_measurement_ticks=spec['measurement_ticks'],
            expected_group_count=len(window), expected_frames=window, raw_count=len(records), first_raw_anomaly_frame=first_bad,
            valid_groups=sum(g['valid'] for g in groups), invalid_groups=sum(not g['valid'] for g in groups),
            group_errors=dict(Counter(e for g in groups for e in g['errors'])), raw_stats=dict(stats),
            inbox_stats={} if inbox is None else dict(inbox.stats), hs_stats={} if hs is None else dict(hs.stats),
            retained_anomaly_images=len(evidence_index), private_metadata_sha256=digest(root/'raw_images_metadata.json'),
            private_snapshots_sha256=digest(root/'snapshots.jsonl') if (root/'snapshots.jsonl').exists() else None,
            tolerance_s=plan['tolerance_s'], sensor_tick_s=0., window_complete=len(window)==spec['measurement_ticks'],
            pixel_capture_association='Not independently verified by static metadata test')
        write(root/'result.json', result)
        print(json.dumps(result, indent=2), flush=True)


def supervise(stage, mode, hypothesis):
    from tools.m8j12_capture_supervisor import listening, group_members
    from tools.m8j13_supervisor import terminate
    import carla
    c = config(); plan = json.loads(PLAN.read_text()); base = Path(plan['private_root']); base.mkdir(parents=True, exist_ok=True)
    spec = plan[stage]; cap = 30+spec['warmup_ticks']+spec['measurement_ticks']+spec['drain_ticks']
    root = base/stage
    # Hold an exclusive ledger lock for the entire server lifetime.
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = base/'budget.json'; ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'runs': []}
        if len(ledger['runs']) >= plan['max_runs'] or sum(r['reserved_ticks'] for r in ledger['runs'])+cap > plan['max_active_ticks']:
            raise RuntimeError('DiagnosticBudgetExhausted')
        if any(listening(p) for p in range(c['port'], c['port']+3)): raise RuntimeError('PortsOccupied')
        root.mkdir(); request = dict(stage=stage, mode=mode, hypothesis=hypothesis, reserved_ticks=cap,
            provenance=provenance(), plan_sha256=digest(PLAN), server_map_boot='Verify map; if needed load Town03 once while asynchronous before sensor creation; no reload/retry')
        write(root/'request.json', request)
        ledger['runs'].append(dict(stage=stage, reserved_ticks=cap, started_monotonic_s=time.monotonic()))
        atomic(ledger_path, ledger)
        server = None; child = None; report = dict(stage=stage, request=request, status='Partial')
        carla_root = Path('/root/gpufree-data/simulators/CARLA_0.9.15')
        binary = carla_root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('DISPLAY', None)
        if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists(): env['VK_ICD_FILENAMES']='/etc/vulkan/icd.d/nvidia_icd.json'
        cmd = ['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--',str(carla_root/'CarlaUE4.sh'),
            '/Game/Carla/Maps/Town03','-RenderOffScreen','-quality-level=Epic','-nosound',
            '-carla-rpc-port='+str(c['port']),'-stdout','-FullStdOutLogOutput']
        report.update(server_argv=cmd, server_binary_sha256=digest(binary))
        try:
            with (root/'server.log').open('x') as log:
                server = subprocess.Popen(cmd, cwd=carla_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+plan['startup_timeout_s']
            while time.monotonic() < deadline:
                if server.poll() is not None: raise RuntimeError('ServerExitedBeforeReady')
                client = carla.Client('127.0.0.1', c['port']); client.set_timeout(2.)
                try:
                    w = client.get_world()
                    report['initial_identity'] = dict(map=w.get_map().name, episode=w.id, client=client.get_client_version(), server=client.get_server_version(), settings=settings_record(w))
                    break
                except RuntimeError: time.sleep(1.)
            else: raise TimeoutError('ServerReadinessTimeout')
            worker_cmd = ['taskset','-c',c['process_cpus']['capture'],sys.executable,'-m',__name__ if __name__!='__main__' else 'tools.m8j13_sync_diagnostic','--worker-root',str(root)]
            report['worker_argv'] = worker_cmd
            with (root/'worker.log').open('x') as log:
                child = subprocess.Popen(worker_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+plan['worker_timeout_s']
                while child.poll() is None:
                    if server.poll() is not None: raise RuntimeError('ServerExitedDuringRun')
                    if time.monotonic() > deadline: raise TimeoutError('WorkerBudgetExceeded')
                    time.sleep(.2)
            report['worker_exit_code'] = child.returncode
            if child.returncode: raise RuntimeError('WorkerFailed')
            report['result'] = json.loads((root/'result.json').read_text()); report['status']='Recorded'
        except BaseException as exc: report['error'] = repr(exc)
        finally:
            terminate(child); terminate(server)
            report['ports_after'] = {str(p):listening(p) for p in range(c['port'],c['port']+3)}
            report['server_group_after'] = [] if server is None else group_members(server.pid)
            report['worker_group_after'] = [] if child is None else group_members(child.pid)
            report['cleanup_complete'] = not any(report['ports_after'].values()) and not report['server_group_after'] and not report['worker_group_after']
            ledger['runs'][-1]['result'] = report.get('result', {}).get('status', 'StartupOrWorkerFailed')
            ledger['runs'][-1]['charged_ticks'] = report.get('result', {}).get('charged_active_ticks', cap)
            atomic(ledger_path, ledger); write(root/'supervisor.json', report)
            write(Path('reports/m8j13_sources/sync_followup')/(stage+'_supervisor.json'), report)
            print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['D1','D2','D3']); p.add_argument('--mode', choices=['paced','barrier']); p.add_argument('--hypothesis'); p.add_argument('--worker-root',type=Path)
    a = p.parse_args(); os.umask(0o077)
    if a.worker_root:
        request = json.loads((a.worker_root/'request.json').read_text())
        # Exclusive sentinel prevents repeated worker attempts under one run ID.
        write(a.worker_root/'worker_started.json', dict(pid=os.getpid(), monotonic_s=time.monotonic()))
        worker(a.worker_root)
    else:
        if not all((a.stage,a.mode,a.hypothesis)): p.error('stage, mode and pre-recorded hypothesis required')
        if a.stage=='D1' and a.mode!='paced': p.error('D1 baseline is paced')
        supervise(a.stage,a.mode,a.hypothesis)


if __name__ == '__main__': main()
