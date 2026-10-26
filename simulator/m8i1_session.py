"""Single bounded TrafficDevelopmentSessionOwner. No HS driving/model imports."""
import hashlib
import json
import random
import socket
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from simulator.synchronizer import FrameSynchronizer
from simulator.m8i1_records import box, pose, xyz, waypoint, physics
from datasets.m8i1_contract import ORDER, MODES, REQUIRED, schedule_frames
from datasets.m8h1_settling import stationary
from geometry.reference import intrinsics_from_hfov, camera_mount_to_hs_from_optical
from tools.m8i1_common import ROOT, PROTOCOL, write, asset, sha, log_json, provenance


class TrafficDevelopmentSessionOwner:
    def __init__(self, start_monotonic):
        self.cfg=json.loads(PROTOCOL.read_text())
        self.started=start_monotonic
        self.actors=[]; self.vehicles=[]; self.walkers=[]; self.controllers=[]; self.sensors=[]
        self.tick_count=0; self.saved=[]; self.events=[]; self.first_sensors={}; self.targets=[]
        self.original=None; self.original_weather=None; self.world=None; self.tm=None; self.ego=None
        self.sync=None; self.cal=[]; self.trace=[]; self.last_positions={}; self.failure=None
        self.paths={k:ROOT/k for k in ['RGBInferencePacket','EvaluatorSidecar']}
        for p in self.paths.values(): p.mkdir()
        self.sensor_events=ROOT/'sensor_events.jsonl'

    def event(self, event, **kwargs):
        row=dict(event=event,wall_elapsed_s=time.monotonic()-self.started,tick=self.tick_count,
                 frame_id=getattr(self,'last_frame',None),**kwargs)
        log_json(ROOT/'events.jsonl',row)
        self.events.append(row)

    def budget(self, reserve=0):
        if time.monotonic()-self.started > self.cfg['wall_seconds']:
            raise RuntimeError('WallBudgetExceeded')
        used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if used+reserve > self.cfg['max_bytes']:
            raise RuntimeError('DiskBudgetExceeded')
        return used

    def connect(self):
        import carla
        self.client=carla.Client('127.0.0.1',self.cfg['port']); self.client.set_timeout(30)
        if self.client.get_client_version()!=self.cfg['server_version'] or self.client.get_server_version()!=self.cfg['server_version']:
            raise RuntimeError('ServerClientVersionMismatch')
        self.world=self.client.get_world(); self.mp=self.world.get_map()
        if self.mp.name.split('/')[-1]!=self.cfg['map']: raise RuntimeError('MapMismatch_NoReload')
        historical=json.loads(Path('reports/m8h0_map_identity_and_runtime.json').read_text())
        xodr=self.mp.to_opendrive()
        if hashlib.sha256(xodr.encode()).hexdigest()!=historical['API_OpenDRIVE']['sha256']:
            raise RuntimeError('OpenDriveMismatch')
        (ROOT/'Town02.xodr').write_text(xodr)
        self.original=self.world.get_settings(); self.original_weather=self.world.get_weather()
        before=self.world.get_actors()
        if self.original.synchronous_mode or any(a.type_id.startswith(('vehicle.','walker.','sensor.','controller.')) for a in before):
            raise RuntimeError('NonExclusiveWorld')
        self.static_actors=[a for a in before if a.type_id!='spectator']
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',self.cfg['tm_port']))==0: raise RuntimeError('TMPortOccupied')
        settings=self.world.get_settings(); settings.synchronous_mode=True
        settings.fixed_delta_seconds=self.cfg['fixed_delta_seconds']; settings.no_rendering_mode=False
        # Keep existing substepping/Physics settings unchanged.
        self.world.apply_settings(settings)
        self.last_frame=self.world.get_snapshot().frame
        time.sleep(.2)
        if self.world.get_snapshot().frame!=self.last_frame: raise RuntimeError('CompetingTickOwner')
        self.tm=self.client.get_trafficmanager(self.cfg['tm_port'])
        self.tm.set_synchronous_mode(True); self.tm.set_random_device_seed(self.cfg['seed'])
        self.tm.set_hybrid_physics_mode(False); self.tm.set_respawn_dormant_vehicles(False)
        self.world.set_pedestrians_seed(self.cfg['walker_seed'])
        self.world.set_weather(getattr(carla.WeatherParameters,self.cfg['weather']))
        write(ROOT/'world_identity.json',dict(server=self.client.get_server_version(),episode=self.world.id,
              map=self.mp.name,xodr=asset(ROOT/'Town02.xodr'),first_owned_frame=self.last_frame,
              original_settings=str(self.original),original_weather=str(self.original_weather),
              actual_settings=str(self.world.get_settings()),weather=str(self.world.get_weather()),
              tm=dict(port=self.cfg['tm_port'],new_exclusive_instance=True,original='NewInstance_DefaultAsync_NoPriorClients',
                      synchronous=True,hybrid=False,respawn=False,seed=self.cfg['seed'])))
        env=[dict(source='EnvironmentObject',entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],
                  **box(e.bounding_box)) for e in self.world.get_environment_objects(carla.CityObjectLabel.Any)]
        write(self.paths['EvaluatorSidecar']/'environment.json',dict(map=self.mp.name,episode=self.world.id,
              objects=env,geometry='SourceEnvironmentOBB_NotMesh'))
        self.event('ExclusiveWorldAndTMEstablished',episode=self.world.id)

    def spawn(self,bp,t,kind,parent=None,**source):
        import carla
        self.budget()
        self.event('SpawnAttempt',kind=kind,blueprint=bp.id,requested_transform=pose(t),source=source)
        kw={} if parent is None else dict(attach_to=parent,attachment_type=carla.AttachmentType.Rigid)
        a=self.world.try_spawn_actor(bp,t,**kw)
        if a is None:
            self.event('SpawnFailed',kind=kind,blueprint=bp.id,source=source)
            return None
        self.actors.append(a)
        self.event('SpawnedActor',kind=kind,actor_id=a.id,blueprint=a.type_id,attributes=dict(a.attributes),
                   requested_transform=pose(t),source=source,
                   local_box=box(a.bounding_box) if kind in ['vehicle','walker'] else None,
                   physics=physics(a) if kind in ['vehicle','walker'] else None)
        return a

    def setup_traffic(self):
        import carla
        lib=self.world.get_blueprint_library(); native=self.mp.get_spawn_points()
        indices=random.Random(self.cfg['seed']).sample(range(len(native)),self.cfg['vehicle_attempts'])
        write(ROOT/'frozen_spawn_order.json',dict(seed=self.cfg['seed'],all_native=[pose(t) for t in native],
              selected_indices=indices,selected_transforms=[pose(native[i]) for i in indices],frozen_before_spawn=True))
        for order,i in enumerate(indices):
            bp=lib.find(self.cfg['vehicle_blueprint']); bp.set_attribute('role_name','m8i1_vehicle')
            a=self.spawn(bp,native[i],'vehicle',native_index=i,attempt_order=order,
                         waypoint=waypoint(self.mp.get_waypoint(native[i].location,project_to_road=False,lane_type=carla.LaneType.Driving)))
            if a is None: continue
            self.vehicles.append(a)
            e=xyz(a.bounding_box.extent)
            if a.type_id!=self.cfg['vehicle_blueprint'] or min(e)<=0 or not np.isfinite(e).all() or a.get_physics_control().mass<=0:
                raise RuntimeError('BasicVehicleIdentityMismatch')
            if self.ego is None:
                self.ego=a; self.event('EgoSelectedBeforeAnyImage',actor_id=a.id,native_index=i)
            a.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            self.event('InitializationHold',actor_id=a.id,control='throttle=0 brake=1 hand_brake=True')
        if self.ego is None: raise RuntimeError('NoSuccessfulVehicle_NoRetry')
        for i in range(self.cfg['walker_attempts']):
            loc=self.world.get_random_location_from_navigation()
            self.event('NavigationReturn',purpose='spawn',request_order=i,xyz=None if loc is None else xyz(loc))
            if loc is None: continue
            bp=lib.find(self.cfg['walker_blueprint'])
            if bp.has_attribute('is_invincible'): bp.set_attribute('is_invincible','false')
            a=self.spawn(bp,carla.Transform(loc),'walker',navigation_request_order=i,original_xyz=xyz(loc))
            if a is not None: self.walkers.append(a)
        for a in self.walkers:
            c=self.spawn(lib.find('controller.ai.walker'),carla.Transform(),'walker_controller',parent=a,walker_id=a.id)
            if c is None: raise RuntimeError('WalkerControllerCreationFailed')
            self.controllers.append((c,a))
        for _ in range(self.cfg['settle_ticks']): self.tick('initial_settle')
        settled=stationary(self.trace,self.cfg['initial_stationarity'])
        write(ROOT/'ego_stationarity.json',dict(decision=settled,trace='ticks.jsonl',physics_never_changed=True))
        if not settled['stable']: raise RuntimeError('EgoNotStationaryBeforeTM')

    def setup_sensors(self):
        import carla
        rig=yaml.safe_load(Path(self.cfg['rig']).read_text())
        assert (rig['image_width'],rig['image_height'],rig['horizontal_fov_deg'],rig['sensor_tick_s'])==(640,384,100,.1)
        self.sync=FrameSynchronizer(REQUIRED,max_frames=20,queue_size=400,
                    timeout_ticks=self.cfg['sensor_timeout_ticks'],cadence_ticks=2,
                    on_event=lambda row:log_json(self.sensor_events,row))
        self.sync.begin_interval(self.last_frame+1)
        lib=self.world.get_blueprint_library()
        modes=dict(rgb='rgb',depth='depth',instance='instance_segmentation',semantic='semantic_segmentation')
        for name in ORDER:
            c=rig['cameras'][name]
            mount=carla.Transform(carla.Location(*c['xyz_m']),carla.Rotation(pitch=c['pitch_deg'],yaw=c['yaw_deg']))
            self.cal.append(dict(name=name,K=intrinsics_from_hfov(640,384,100).tolist(),
                                 T_cam_to_ego=camera_mount_to_hs_from_optical(mount.get_matrix()).tolist(),
                                 mount_matrix=mount.get_matrix()))
            for kind in MODES:
                bp=lib.find('sensor.camera.'+modes[kind])
                for k,v in dict(image_size_x=640,image_size_y=384,fov=100,sensor_tick=.1,lens_k=0,lens_kcube=0).items():
                    bp.set_attribute(k,str(v))
                if kind=='rgb': bp.set_attribute('motion_blur_intensity','0')
                a=self.spawn(bp,mount,'sensor',parent=self.ego,camera=name,modality=kind)
                if a is None: raise RuntimeError('SensorSpawnFailed')
                self.sensors.append(a); sensor_id=f'{name}/{kind}'; callback=self.sync.callback(sensor_id)
                def receive(image,n=sensor_id,cb=callback):
                    if n not in self.first_sensors: self.first_sensors[n]=dict(frame=image.frame,timestamp=image.timestamp)
                    cb(image)
                a.listen(receive)
                self.event('SensorListening',actor_id=a.id,sensor_id=sensor_id,actual_attributes=dict(a.attributes))
        write(ROOT/'calibration.json',dict(rig=asset(self.cfg['rig']),cameras=self.cal,width=640,height=384,hfov=100,sensor_tick=.1))
        while self.tick_count<self.cfg['init_ticks']: self.tick('sensor_warmup')
        if set(self.first_sensors)!=set(REQUIRED): raise RuntimeError('MissingSensorWarmup')
        phases={r['frame']%2 for r in self.first_sensors.values()}
        if len(phases)!=1: raise RuntimeError('SensorCadencePhaseMismatch')
        first=min(r['frame'] for r in self.first_sensors.values())
        self.targets=schedule_frames(first,self.last_frame,self.cfg['save_offsets_sensor_periods'])
        write(ROOT/'frozen_save_schedule.json',dict(first_sensor_records=self.first_sensors,
              phase=first%2,source='ActualSensorPhase_NotAssumedParity',last_init_frame=self.last_frame,
              requested_frames=self.targets,frozen_before_TM_observation=True))
        for a in self.vehicles:
            a.apply_control(carla.VehicleControl()); a.set_autopilot(True,self.cfg['tm_port'])
            self.event('TMControlledVehicle',actor_id=a.id,enabled=True,physics_changes=0)
        for i,(controller,walker) in enumerate(self.controllers):
            loc=self.world.get_random_location_from_navigation()
            self.event('NavigationReturn',purpose='destination',request_order=i,xyz=None if loc is None else xyz(loc))
            if loc is None: self.event('WalkerDestinationUnavailable',actor_id=walker.id); continue
            controller.start(); controller.go_to_location(loc); controller.set_max_speed(1.4)
            self.event('WalkerAIControlled',actor_id=walker.id,controller_id=controller.id,destination=xyz(loc),speed_mps=1.4)

    def records(self,snap):
        import carla
        all_actors=self.world.get_actors()
        unexpected=[a.id for a in all_actors if a.type_id.startswith(('vehicle.','walker.','sensor.','controller.')) and a.id not in {v.id for v in self.actors}]
        if unexpected: raise RuntimeError('UnownedActors:'+str(unexpected))
        rows=[]
        for a in all_actors:
            if a.type_id.startswith(('sensor.','controller.','spectator')): continue
            s=snap.find(a.id)
            if s is None: raise RuntimeError('ActorSnapshotMissing:'+str(a.id))
            t=s.get_transform(); owned=a.id in {v.id for v in self.vehicles+self.walkers}
            row=dict(source='Actor',entity_id=a.id,name=a.type_id,semantic_tags=list(a.semantic_tags),
                     pose=pose(t),entity_transform=t.get_matrix(),velocity_mps=xyz(s.get_velocity()),
                     acceleration_mps2=xyz(s.get_acceleration()),angular_velocity_deg_s=xyz(s.get_angular_velocity()),
                     frame_id=snap.frame,timestamp_s=snap.timestamp.elapsed_seconds,snapshot_present=True,
                     is_ego=a.id==self.ego.id,owned_traffic=owned,**box(a.bounding_box,t))
            if owned:
                row['control']=str(a.get_control())
                row['waypoint']=waypoint(self.mp.get_waypoint(t.location,project_to_road=False,lane_type=carla.LaneType.Any))
                previous=self.last_positions.get(a.id)
                lim=self.cfg['severe_state']
                if max(abs(t.rotation.pitch),abs(t.rotation.roll))>lim['max_abs_tilt_deg'] or np.linalg.norm(row['velocity_mps'])>lim['max_speed_mps']:
                    raise RuntimeError('SevereStateAnomaly:'+str(a.id))
                if previous is not None and abs(t.location.z-previous[2])>lim['max_abs_z_change_per_tick_m']:
                    raise RuntimeError('UnexplainedVerticalJump:'+str(a.id))
                self.last_positions[a.id]=xyz(t.location)
            rows.append(row)
        expected={a.id for a in self.vehicles+self.walkers}
        if expected!={r['entity_id'] for r in rows if r['owned_traffic']}: raise RuntimeError('UncontrolledActorLifecycle')
        return rows

    def tick(self,phase):
        if self.tick_count>=self.cfg['max_ticks']: raise RuntimeError('TickBudgetExceeded')
        self.budget()
        if self.world.get_snapshot().frame!=self.last_frame: raise RuntimeError('ForeignWorldAdvance')
        before=time.monotonic(); frame=self.world.tick(30); snap=self.world.get_snapshot(); self.tick_count+=1
        if snap.frame!=frame or frame!=self.last_frame+1: raise RuntimeError('TickGap')
        self.last_frame=frame; objects=self.records(snap)
        ego=next(r for r in objects if r['is_ego']); self.trace.append(ego)
        log_json(ROOT/'ticks.jsonl',dict(tick=self.tick_count,phase=phase,frame_id=frame,
            timestamp_s=snap.timestamp.elapsed_seconds,delta_seconds=snap.timestamp.delta_seconds,
            tick_wall_s=time.monotonic()-before,objects=[r for r in objects if r['owned_traffic']]))
        if self.sync:
            self.sync.add_snapshot(frame,snap.timestamp.elapsed_seconds,ego,objects)
            end=time.monotonic()+self.cfg['sensor_drain_wall_s']
            while True:
                for bundle in self.sync.drain():
                    if bundle.frame_id in self.targets: self.save(bundle)
                if time.monotonic()>=end: break
                time.sleep(.01)
            if phase=='observation':
                if any(f < frame-self.cfg['sensor_timeout_ticks'] and f not in self.saved for f in self.targets):
                    raise RuntimeError('ScheduledFrameMissing_NoNearestFill')
                if self.sync.stats['queue_overflow'] or self.sync.stats['invalid_frame']:
                    raise RuntimeError('CriticalSensorIntegrityFailure')
        else: time.sleep(self.cfg['tick_wall_delay'])

    def save(self,b):
        import carla
        self.budget(reserve=32*1024*1024)
        if b.frame_id in self.saved or len(self.saved)>=12: raise RuntimeError('DuplicateOrExcessKeyframe')
        E=np.asarray(b.ego['entity_transform']); sensors={}; closure=[]
        packet=dict(schema='RGBInferencePacket.m8i1.v1',source='CARLA0.9.15_RGBOnly',
             calibration_sha256=sha(ROOT/'calibration.json'),camera_order=ORDER,image_size=[384,640],
             frame_id=b.frame_id,timestamp_s=b.timestamp_s,cameras=[],image_transform='Identity')
        calibration=[]
        for c in self.cal:
            name=c['name']; expected=E@np.asarray(c['mount_matrix']); sr={}
            calibration.append(dict(c,T_world_from_camera=expected.tolist()))
            for kind in MODES:
                sample=b.sensors[f'{name}/{kind}']
                delta=float(np.max(np.abs(sample.T_carla_world_from_camera-expected)))
                closure.append(dict(camera=name,modality=kind,matrix_max_abs_error=delta,
                                    passed=delta<=self.cfg['camera_closure_matrix_atol']))
                if not closure[-1]['passed']:
                    write(ROOT/f'failed_closure_{b.frame_id}_{name}_{kind}.json',dict(expected=expected.tolist(),actual=sample.T_carla_world_from_camera.tolist(),error=delta))
                    raise RuntimeError('CameraPoseClosureFailed')
                arr=np.frombuffer(sample.raw_bgra,np.uint8).reshape(384,640,4)
                directory=self.paths['RGBInferencePacket' if kind=='rgb' else 'EvaluatorSidecar']
                path=directory/f'{b.frame_id:06}_{name}_{kind}.png'
                if path.exists(): raise FileExistsError(path)
                if not cv2.imwrite(str(path),arr[:,:,:3] if kind=='rgb' else arr): raise RuntimeError('PNGWriteFailed')
                sr[kind]=dict(**asset(path),raw_bgra_sha256=hashlib.sha256(sample.raw_bgra).hexdigest(),
                    frame_id=sample.frame_id,timestamp_s=sample.timestamp_s,transform=sample.T_carla_world_from_camera.tolist())
            sensors[name]=sr
            packet['cameras'].append(dict(name=name,path=Path(sr['rgb']['path']).name,sha256=sr['rgb']['sha256'],
                 K=c['K'],T_cam_to_ego=c['T_cam_to_ego'],frame_id=b.frame_id,timestamp_s=b.timestamp_s))
        write(self.paths['RGBInferencePacket']/f'{b.frame_id:06}_packet.json',packet)
        region=[]
        for obj in b.objects:
            if not obj['owned_traffic']: continue
            corners=np.asarray(obj['corners_world_m'])[[0,4,6,2]]
            checks=[]
            for point in corners:
                wp=self.mp.get_waypoint(carla.Location(*point),project_to_road=False,lane_type=carla.LaneType.Any)
                checks.append(dict(point_world=point.tolist(),waypoint=waypoint(wp)))
            region.append(dict(actor_id=obj['entity_id'],footprint_corner_map_queries=checks,
                               evidence='AuthoredStaticMapAtSnapshotCoordinates_NotMeshContact'))
        side=dict(schema='EvaluatorSidecar.m8i1.v1',source='CARLA_Privileged_EvaluatorOnly',world_episode=self.world.id,
             map=self.mp.name,split='DevelopmentOnly',frame_id=b.frame_id,timestamp_s=b.timestamp_s,
             T_carla_world_from_ego=E.tolist(),objects=b.objects,sensors=sensors,calibration=calibration,
             camera_pose_closure=closure,region_evidence=region,
             environment_inventory=asset(self.paths['EvaluatorSidecar']/'environment.json'))
        write(self.paths['EvaluatorSidecar']/f'{b.frame_id:06}_sidecar.json',side)
        self.saved.append(b.frame_id); self.event('KeyframeSaved',source_frame=b.frame_id,payload_count=24)
        print('saved',b.frame_id,'keyframes',len(self.saved),flush=True)

    def run(self):
        try:
            self.connect(); self.setup_traffic(); self.setup_sensors()
            for _ in range(self.cfg['observation_ticks']): self.tick('observation')
            if self.saved!=self.targets: raise RuntimeError('IncompleteSchedule')
        except Exception as exc:
            self.failure=type(exc).__name__+': '+str(exc); self.event('PilotStopped',reason=self.failure)
            print(self.failure,flush=True)
        finally:
            self.cleanup()
            write(ROOT/'session_result.json',dict(**provenance('ActualCARLATrafficSession'),
                failure=self.failure,ticks=self.tick_count,requested_frames=self.targets,saved_frames=self.saved,
                first_sensor_records=self.first_sensors,synchronizer_stats=dict(self.sync.stats) if self.sync else {},
                vehicle_ids=[a.id for a in self.vehicles],walker_ids=[a.id for a in self.walkers],
                ego_id=None if self.ego is None else self.ego.id,wall_elapsed_s=time.monotonic()-self.started,
                data_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())))

    def cleanup(self):
        rows=[]
        def attempt(action,fn,actor_id=None):
            try:
                result=fn(); rows.append(dict(action=action,actor_id=actor_id,result=str(result),ok=result is not False))
            except Exception as exc: rows.append(dict(action=action,actor_id=actor_id,ok=False,error=str(exc)))
        for a in self.sensors: attempt('stop_sensor',a.stop,a.id)
        if self.sync:
            self.sync.finish()
        for c,a in self.controllers: attempt('stop_walker_controller',c.stop,c.id)
        for a in self.vehicles:
            if self.tm: attempt('TM_unregister',lambda a=a:a.set_autopilot(False,self.cfg['tm_port']),a.id)
        for a in reversed(self.actors): attempt('destroy',a.destroy,a.id)
        if self.tm: attempt('TM_restore_default_async',lambda:self.tm.set_synchronous_mode(False))
        if self.world is not None:
            try:
                remaining={a.id for a in self.world.get_actors()} & {a.id for a in self.actors}
                rows.append(dict(action='verify_owned_actor_absence',remaining_ids=sorted(remaining),ok=not remaining))
            except Exception as exc: rows.append(dict(action='verify_owned_actor_absence',ok=False,error=str(exc)))
        if self.original_weather is not None: attempt('restore_weather',lambda:self.world.set_weather(self.original_weather))
        if self.original is not None:
            attempt('restore_world_settings',lambda:self.world.apply_settings(self.original))
            attempt('verify_world_settings',lambda:str(self.world.get_settings())==str(self.original))
            attempt('verify_weather',lambda:str(self.world.get_weather())==str(self.original_weather))
        write(ROOT/'cleanup.json',dict(rows=rows,all_operations_ok=all(r['ok'] for r in rows),
              physics_changes=0,cleanup_ticks=0,server_shutdown='Supervisor responsibility'))
