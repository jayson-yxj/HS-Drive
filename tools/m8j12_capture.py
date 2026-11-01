"""Two fixed, bounded official-camera observations; existing Oracle collector only.

This process cannot import the new model or consume its predictions. CARLA is
owned by a separate bounded supervisor and must be stopped before model replay.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
from types import SimpleNamespace
import time

import numpy as np
from simulator.session import CarlaSession
from simulator.synchronizer import FrameSynchronizer
from simulator.oracle_driver import OracleState, PurePursuitPID, local_trajectory
from datasets.m8j12_contract import CAMERAS, official_camera_contract, assert_observation


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2300);p.add_argument('--attempt',default='first');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    os.umask(0o077)
    c=json.loads(Path('configs/m8j12_protocol.json').read_text())
    plan=json.loads(Path('configs/m8j12_capture.json').read_text())
    root=Path(c['private_root'])/('observations' if a.attempt=='first' else 'observations_'+a.attempt)
    source=Path(c['runtime_root'])/'official-source'
    camera=official_camera_contract(source/'leaderboard/team_code/sparsedrive_b2d_agent.py')
    spec=importlib.util.spec_from_file_location('native_route',source/'leaderboard/team_code/planner.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.RoutePlanner(4.,50.)  # preflight includes its actual Pillow-backed constructor
    assert world_transform_smoke(carla)
    if a.preflight_only:
        print('Capture imports, native navigation and CARLA transform preflight passed');return
    root.mkdir()
    reports=[]
    for seq in plan['sequences']:
        root_seq=root/seq['id'];root_seq.mkdir()
        raw=json.loads(Path(seq['route_file']).read_text())
        route=np.array(raw['route_world_hs']);world_route=np.array(raw['route_world_carla'])
        start=seq['start_index'];route=route[start:];world_route=world_route[start:]
        cfg=dict(host='127.0.0.1',port=a.port,client_timeout_s=45,map='Town03',fixed_delta_seconds=.1)
        with CarlaSession(cfg,load_map=True) as session:
            world=session.world;cmap=world.get_map();world.set_weather(carla.WeatherParameters.ClearNoon)
            write(root_seq/'identity.json',dict(map=cmap.name,episode=world.id,
                xodr_sha256=hashlib.sha256(cmap.to_opendrive().encode()).hexdigest(),
                client=session.client.get_client_version(),server=session.client.get_server_version()))
            xyz=world_route[0];tf=carla.Transform(carla.Location(*map(float,xyz)),carla.Rotation(yaw=float(-np.degrees(route[0,2]))));tf.location.z+=.5
            ego=session.spawn(world.get_blueprint_library().find(plan['blueprint']),tf)
            imu_queue=Queue(400);imu_pending={};states={};controls={}
            sync=FrameSynchronizer(CAMERAS,cadence_ticks=1,max_frames=20,timeout_ticks=20)
            for name in CAMERAS:
                s=camera['sensors'][name];bp=world.get_blueprint_library().find('sensor.camera.rgb')
                for key,val in [('image_size_x',704),('image_size_y',396),('fov',s['fov']),('sensor_tick',.1)]:bp.set_attribute(key,str(val))
                cam=session.spawn(bp,carla.Transform(carla.Location(x=s['x'],y=s['y'],z=s['z']),carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll'])),parent=ego)
                cam.listen(sync.callback(name))
            bp=world.get_blueprint_library().find('sensor.other.imu');bp.set_attribute('sensor_tick','0.05')
            imu=session.spawn(bp,carla.Transform(carla.Location(x=-1.4)),parent=ego);imu.listen(imu_queue.put_nowait)
            nav=mod.RoutePlanner(4.,50.)
            nav.set_route([(SimpleNamespace(location=SimpleNamespace(x=float(x[0]),y=float(x[1]))),SimpleNamespace(value=4)) for x in world_route])
            controller=PurePursuitPID();rows=[];pending_bundles={};begin=None;sim_start=world.get_snapshot().timestamp.elapsed_seconds
            # spawn_actor does not add the actor to the *previous* WorldSnapshot.
            # Establish one measured synchronous snapshot before querying ego state.
            world.tick()
            for tick in range(plan['settle_ticks']+plan['drive_warmup_ticks']+32+10):
                snap=world.get_snapshot();actor=snap.find(ego.id);t=actor.get_transform();v=actor.get_velocity();speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
                state=OracleState(t.location.x,-t.location.y,-math.radians(t.rotation.yaw),speed,snap.timestamp.elapsed_seconds)
                if tick<plan['settle_ticks']:command=dict(steer=0.,throttle=0.,brake=1.)
                else:
                    trajectory,_=local_trajectory(route,state,max_speed=plan['speed_mps'])
                    command=controller.command(trajectory,state,dt=.1)
                # ONLY the existing collection controller reaches actuation.
                ego.apply_control(carla.VehicleControl(steer=command['steer'],throttle=command['throttle'],brake=command['brake']))
                f=world.tick();snap=world.get_snapshot();actor=snap.find(ego.id);t=actor.get_transform();v=actor.get_velocity()
                if snap.frame!=f:raise RuntimeError('CompetingTickOwner')
                if snap.timestamp.elapsed_seconds-sim_start>20:raise RuntimeError('SimulationBudgetExceeded')
                first_tick=plan['settle_ticks']+plan['drive_warmup_ticks']
                if tick<first_tick:
                    while not sync.queue.empty():sync.queue.get_nowait()
                    while not imu_queue.empty():imu_queue.get_nowait()
                    continue
                if begin is None:begin=f;sync.begin_interval(f,explicit_cadence=True)
                target,command_nav=nav.run_step(np.array([t.location.x,t.location.y]))
                forward=t.get_forward_vector()
                row=dict(frame_id=f,timestamp_s=snap.timestamp.elapsed_seconds,sequence_id=seq['id'],
                    ego_world_matrix=t.get_matrix(),speed_mps=v.x*forward.x+v.y*forward.y+v.z*forward.z,
                    speed_source='native speedometer signed forward projection of measured velocity',
                    navigation_command=command_nav.value,target_world_xy=target.tolist(),
                    collection_control=command,ego_state_source='Oracle Ego State',
                    target_source='fixed historical navigation route; original RoutePlanner(4,50)',model_control_applied=0)
                states[f]=row;sync.add_snapshot(f,row['timestamp_s'],{},[])
                deadline=time.monotonic()+1.
                while time.monotonic()<deadline:
                    while True:
                        try:x=imu_queue.get_nowait()
                        except Empty:break
                        if x.frame>=begin:imu_pending[x.frame]=x
                    for bundle in sync.drain():pending_bundles[bundle.frame_id]=bundle
                    expected=begin+len(rows)
                    if expected in pending_bundles and expected in imu_pending:
                        bundle=pending_bundles.pop(expected);m=imu_pending.pop(expected);row=states.pop(expected)
                        if abs(m.timestamp-row['timestamp_s'])>1e-5:raise RuntimeError('IMUTimeMismatch')
                        row['imu']=[m.accelerometer.x,m.accelerometer.y,m.accelerometer.z,m.gyroscope.x,m.gyroscope.y,m.gyroscope.z,m.compass]
                        row['imu_frame']=m.frame;row['imu_timestamp_s']=m.timestamp
                        images={k:np.frombuffer(s.raw_bgra,np.uint8).reshape(396,704,4)[:,:,:3].copy() for k,s in bundle.sensors.items()}
                        row['camera_frames']={k:s.frame_id for k,s in bundle.sensors.items()};row['camera_timestamps_s']={k:s.timestamp_s for k,s in bundle.sensors.items()}
                        row['camera_world_matrices']={k:s.T_carla_world_from_camera.tolist() for k,s in bundle.sensors.items()}
                        assert_observation(row,images)
                        row['images_file']=str(expected)+'.npz';np.savez_compressed(root_seq/row['images_file'],**images)
                        rows.append(row)
                        if len(rows)==32:break
                    else:time.sleep(.005)
                    if len(rows)==32:break
                if len(rows)==32:break
            if len(rows)!=32:raise RuntimeError('FixedConsecutiveWindowIncomplete')
            write(root_seq/'manifest.json',rows)
            yaw=np.unwrap([math.atan2(r['ego_world_matrix'][1][0],r['ego_world_matrix'][0][0]) for r in rows])
            reports.append(dict(sequence=seq['id'],frames=len(rows),first_frame=rows[0]['frame_id'],last_frame=rows[-1]['frame_id'],
                span_s=rows[-1]['timestamp_s']-rows[0]['timestamp_s'],speed_range=[min(r['speed_mps'] for r in rows),max(r['speed_mps'] for r in rows)],
                yaw_change_rad=float(yaw[-1]-yaw[0]),simulation_elapsed_s=world.get_snapshot().timestamp.elapsed_seconds-sim_start,
                sync_stats=dict(sync.stats),manifest_sha256=hashlib.sha256((root_seq/'manifest.json').read_bytes()).hexdigest()))
    write(Path('reports/m8j12_sources/capture_result.json'),dict(status='Captured',observation_root=str(root),sequences=reports,model_control_applied=0))


def world_transform_smoke(carla):
    t=carla.Transform(carla.Location(*map(float,[1,2,3])),carla.Rotation(yaw=30.))
    return np.isfinite(t.get_matrix()).all()


if __name__=='__main__':main()
