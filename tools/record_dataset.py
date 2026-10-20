"""Capture stationary or low-speed scripted geometry sequences (Oracle/Privileged)."""
import argparse
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
from datasets.schema import CAMERA_IDS
from simulator.session import CarlaSession
from simulator.camera_rig import spawn_rig
from simulator.imaging import load_camera_config
from simulator.synchronizer import FrameSynchronizer
from simulator.snapshot import extract_snapshot
from simulator.recorder import Recorder
from tools.provenance import source_hashes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-config', help='Versioned collector defaults; explicit CLI options override')
    parser.add_argument('--config', default='configs/carla.yaml')
    parser.add_argument('--cameras', default='configs/cameras.yaml')
    parser.add_argument('--render-every-tick', action=argparse.BooleanOptionalAction, default=False, help='Render each physics step, retain global 10Hz frame cadence')
    parser.add_argument('--camera-profile', help='Optional versioned imaging profile')
    parser.add_argument('--output', default='data/raw')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--load-map', action='store_true')
    parser.add_argument('--front-only', action='store_true', help='M0 RGB/depth smoke test')
    parser.add_argument('--vehicle', default='vehicle.tesla.model3')
    parser.add_argument('--spawn-index', type=int, default=0)
    parser.add_argument('--target-distance', type=float, default=None)
    parser.add_argument('--target-lateral-speed', type=float, default=0.0, help='Scripted target motion in CARLA ego +Y, m/s')
    parser.add_argument('--motion', choices=['stationary','straight','autopilot'], default='stationary')
    parser.add_argument('--map', help='Explicit map override, requires --load-map if different')
    parser.add_argument('--seed', type=int, help='Simulator / Traffic Manager random seed')
    parser.add_argument('--tm-port', type=int, default=8050, help='Exclusive unused Traffic Manager port')
    parser.add_argument('--sun-altitude', type=float, help='Controlled daylight weather override')
    parser.add_argument('--traffic-rules',action='store_true',help='Capture same-tick privileged signal supervision')
    parser.add_argument('--signal-probe',action='store_true',help='Stationary controlled red/yellow/green diagnostic; enables traffic rules')
    parser.add_argument('--throttle', type=float, default=0.2)
    parser.add_argument('--speed-limit', type=float, default=2.0)
    parser.add_argument('--render-quality', choices=['Low','Epic'], default='Epic', help='Record actual server launch quality; does not change the server')
    parser.add_argument('--max-ticks', type=int, default=None)
    parser.add_argument('--tick-wall-delay', type=float, default=0.3)
    parser.add_argument('--warmup-ticks', type=int, default=20)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.capture_config:
        from simulator.capture_config import load_capture_defaults
        parser.set_defaults(**load_capture_defaults(preliminary.capture_config))
    args = parser.parse_args(argv)
    if args.tick_wall_delay < 0 or args.warmup_ticks < 0:
        parser.error('Delay and warmup must be nonnegative')
    if not 0 <= args.throttle <= 0.4 or not 0 < args.speed_limit <= 5:
        parser.error('Geometry trial requires throttle in [0,.4] and speed limit in (0,5]')
    if abs(args.target_lateral_speed)>1 or (args.target_lateral_speed and args.target_distance is None):
        parser.error('Target lateral motion requires target-distance and abs(speed)<=1m/s')
    if args.motion != 'stationary' and args.target_distance is not None:
        parser.error('Moving ego trial cannot use a stationary target ahead')
    if args.signal_probe and (args.motion!='stationary' or args.target_distance is not None):parser.error('Signal probe requires stationary ego and no target')
    if args.frames < 1:
        parser.error('--frames must be positive')
    return args


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cameras = load_camera_config(args.cameras,args.camera_profile)
    if args.map:
        cfg['map'] = args.map
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.render_every_tick:
        cameras['render_sensor_tick_s'] = 0.0
    if args.host:
        cfg['host'] = args.host
    if args.port:
        cfg['port'] = args.port
    ratio = cameras['sensor_tick_s'] / cfg['fixed_delta_seconds']
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('sensor_tick must be an integer multiple of fixed_delta_seconds')
    names = ('front',) if args.front_only else CAMERA_IDS
    required = tuple(f'{c}/{k}' for c in names for k in ('rgb', 'depth'))
    cap = cfg['capture']
    sync = FrameSynchronizer(required, max_frames=cap['max_buffered_frames'],
        queue_size=cap['sensor_queue_max_size'], timeout_ticks=cap['frame_timeout_ticks'],
        cadence_ticks=round(ratio))
    recorder = None
    status = 'failed'
    start = time.monotonic()
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run-id must be a simple directory name')
    def interrupt(*_):
        raise KeyboardInterrupt
    old_signal = signal.signal(signal.SIGTERM, interrupt)
    try:
        with CarlaSession(cfg, load_map=args.load_map) as session:
            import carla
            if args.sun_altitude is not None:
                weather = session.world.get_weather()
                weather.sun_altitude_angle = args.sun_altitude
                session.world.set_weather(weather)
            spawns = session.world.get_map().get_spawn_points()
            if not 0 <= args.spawn_index < len(spawns):
                raise ValueError('spawn-index outside map spawn points')
            bp = session.world.get_blueprint_library().find(args.vehicle)
            bp.set_attribute('role_name', 'hsdrive_oracle')
            probe=None;traffic=None;spawn=spawns[args.spawn_index]
            if args.signal_probe:
                from simulator.traffic_rules import choose_probe
                probe,spawn=choose_probe(session.world)
            if args.traffic_rules or args.signal_probe:
                from simulator.traffic_rules import TrafficRules
                traffic=TrafficRules(session.world,probe=probe);session.cleanup_callbacks.append(traffic.close)
            ego = session.spawn(bp, spawn)
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            # Settle suspension before defining the capture interval.
            for _ in range(20):
                session.world.tick()
            target = None
            if args.target_distance is not None:
                if args.target_distance < 6:
                    raise ValueError('Target distance must be >=6m')
                pose = ego.get_transform()
                location = pose.transform(carla.Location(x=args.target_distance, z=0.5))
                target = session.spawn(bp, carla.Transform(location, pose.rotation))
                target.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                for _ in range(20):
                    session.world.tick()
            if target is not None and args.target_lateral_speed:
                target.set_simulate_physics(False)
                target_start = target.get_transform()
                target_direction = ego.get_transform().get_right_vector()
            calibration = spawn_rig(session, ego, cameras, sync, names)
            # Render warmup is outside the recorded interval. Discard callbacks
            # explicitly, before any WorldSnapshot is admitted to the join.
            from queue import Empty
            warmup_samples = 0
            for _ in range(args.warmup_ticks):
                session.world.tick()
                time.sleep(args.tick_wall_delay)
                while True:
                    try:
                        sync.queue.get_nowait()
                        warmup_samples += 1
                    except Empty:
                        break
            if args.motion == 'autopilot':
                session.start_autopilot(ego, port=args.tm_port, speed_mps=args.speed_limit)
            sync.begin_interval(session.world.get_snapshot().frame + 1, explicit_cadence=args.render_every_tick)
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
            weather = session.world.get_weather()
            weather_fields = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
                'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance', 'wetness')
            metadata = {'run_id': run_id, 'carla_version': session.client.get_server_version(),
                'git_sha': result.stdout.strip() if result.returncode == 0 else None,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'map': session.world.get_map().name, 'vehicle': args.vehicle,
                'spawn_index': args.spawn_index,
                'traffic_rules_enabled':traffic is not None,'signal_probe':args.signal_probe,
                'seed': cfg['seed'], 'config': cfg, 'camera_config': cameras,
                'source_sha256': source_hashes(),
                'weather': {key: getattr(weather, key) for key in weather_fields},
                'capture_policy': 'owned_traffic_manager_random_route' if args.motion == 'autopilot' else args.motion + '_ego_no_traffic_manager',
                'tm_port': args.tm_port if args.motion == 'autopilot' else None,
                'navigation_label': 'not exported; route_command remains unknown',
                'motion': args.motion, 'throttle': args.throttle, 'speed_limit_mps': args.speed_limit,
                'render_quality_declared': args.render_quality,
                'capture_defaults_file': args.capture_config,
                'target_distance_m': args.target_distance,
                'target_lateral_speed_mps': args.target_lateral_speed,
                'target_actor_id': target.id if target else None,
                'target_motion_policy': 'kinematic_transform' if args.target_lateral_speed else 'static',
                'tick_wall_delay_s': args.tick_wall_delay,
                'warmup_ticks': args.warmup_ticks, 'warmup_samples_discarded': warmup_samples,
                'depth_convention': 'unverified',
                'dependencies': subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze'], text=True)}
            recorder = Recorder(Path(args.output)/run_id, calibration, metadata)
            # OpenDRIVE is privileged supervision provenance, never a model input.
            from simulator.recorder import file_hash, atomic_json
            map_path = recorder.root/'map.xodr'
            map_path.write_text(session.world.get_map().to_opendrive())
            atomic_json(recorder.root/'map_provenance.json', {'schema_version':'hsd-map-source-v1',
                'map':session.world.get_map().name, 'opendrive_sha256':file_hash(map_path),
                'carla_version':session.client.get_server_version()})
            sync.on_event = recorder.event
            # Actor bounding boxes are static local geometry; poses come from snapshots.
            catalog = {a.id: (a.type_id, a.bounding_box) for a in session.world.get_actors()
                       if a.type_id.startswith(('vehicle.', 'walker.'))}
            max_ticks = args.max_ticks or (args.frames * round(ratio) + 4*cap['frame_timeout_ticks'])
            for tick_index in range(max_ticks):
                if probe is not None:traffic.set_probe_phase(tick_index)
                if target is not None and args.target_lateral_speed:
                    offset = args.target_lateral_speed * cfg['fixed_delta_seconds'] * tick_index
                    target.set_transform(carla.Transform(
                        target_start.location + target_direction * offset, target_start.rotation))
                if args.motion == 'straight':
                    velocity = ego.get_velocity()
                    speed = np.linalg.norm([velocity.x,velocity.y,velocity.z])
                    ego.apply_control(carla.VehicleControl(
                        throttle=args.throttle if speed < args.speed_limit else 0.0,
                        brake=0.15 if speed > args.speed_limit + 0.2 else 0.0))
                frame = session.world.tick()
                snapshot = session.world.get_snapshot()
                if snapshot.frame != frame:
                    raise RuntimeError('Snapshot/tick mismatch: competing tick owner')
                state, objects = extract_snapshot(snapshot, ego, catalog)
                rules=traffic.capture(snapshot,ego) if traffic is not None else None
                sync.add_snapshot(frame, snapshot.timestamp.elapsed_seconds, state, objects,rules)
                # Give callback threads a bounded scheduling window; never block waiting
                # for a frame whose renderer needs a later tick to complete.
                time.sleep(args.tick_wall_delay)
                for bundle in sync.drain():
                    if recorder.committed < args.frames:
                        recorder.write(bundle)
                if recorder.committed >= args.frames:
                    status = 'complete'
                    break
            if status != 'complete':
                raise TimeoutError(f'Only {recorder.committed}/{args.frames} frames in {max_ticks} ticks')
            for sensor in session.sensors:
                sensor.stop()
            session.sensors.clear()
            for bundle in sync.finish():
                if recorder.committed < args.frames:
                    recorder.write(bundle)
            print(json.dumps({'scene': str(recorder.root), 'committed': recorder.committed,
                              'wall_seconds': time.monotonic()-start, 'sync': dict(sync.stats)}, indent=2))
    except KeyboardInterrupt:
        status = 'interrupted'
        return 130
    except Exception as exc:
        if recorder:
            recorder.event({'reason': 'fatal', 'detail': repr(exc)})
        print(f'Capture failed: {exc}', file=__import__('sys').stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        if recorder:
            if status != 'complete':
                sync.finish()  # report all incomplete buffers after session cleanup
            recorder.close(status, sync.stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
