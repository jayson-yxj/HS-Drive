"""J13 tick owner, same-frame sensors and checked actuation. No model imports."""
import importlib.util
import math
from pathlib import Path
from queue import Queue,Empty,Full
from types import SimpleNamespace
import time
import numpy as np
from simulator.session import CarlaSession
from simulator.synchronizer import FrameSynchronizer
from simulator.m8j13_sensor_audit import RawInbox
from datasets.m8j12_contract import CAMERAS,official_camera_contract,assert_observation


class J13Session(CarlaSession):
    def __enter__(self):
        timeout=self.config['client_timeout_s'];self.config=dict(self.config,client_timeout_s=45.)
        super().__enter__()
        # Every repetition starts from a reload with the same synchronous settings.
        try:
            self.world=self.client.reload_world(False)
            actual=self.world.get_settings()
            if not actual.synchronous_mode or actual.fixed_delta_seconds is None or abs(actual.fixed_delta_seconds-self.config['fixed_delta_seconds'])>1e-9 or actual.no_rendering_mode:
                raise RuntimeError('ReloadedWorldSettingsMismatch')
            self.config['client_timeout_s']=timeout
            self.client.set_timeout(timeout)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise


class Actuator:
    def __init__(self,client,ego):self.client=client;self.ego=ego;self.inject_failure=False
    def apply(self,control):
        import carla
        if self.inject_failure:raise RuntimeError('InjectedPrimaryControlConnectionFailure')
        self.ego.apply_control(carla.VehicleControl(**control))

    def verify(self,expected):
        actual=self.ego.get_control()
        if any(abs(getattr(actual,k)-expected[k])>1e-5 for k in ('steer','throttle','brake')):
            raise RuntimeError('ActuationFeedbackMismatch')
        return actual


def measurement(snapshot,ego):
    actor=snapshot.find(ego.id)
    if actor is None:raise RuntimeError('EgoAbsentFromSnapshot')
    tf=actor.get_transform();v=actor.get_velocity();forward=tf.get_forward_vector()
    return dict(frame=snapshot.frame,sim_s=snapshot.timestamp.elapsed_seconds,pose=tf.get_matrix(),
        speed=v.x*forward.x+v.y*forward.y+v.z*forward.z,
        speed_norm=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),velocity=[v.x,v.y,v.z],
        position=[tf.location.x,tf.location.y,tf.location.z],yaw=math.radians(tf.rotation.yaw))


def spawn_ego(session,c,scene):
    import carla,json
    raw=json.loads(Path(scene['route_file']).read_text())
    xyz=np.array(raw['route_world_carla'])[scene['start_index']]
    route=np.array(raw['route_world_hs'])[scene['start_index']:scene['end_index']+1]
    world_route=np.array(raw['route_world_carla'])[scene['start_index']:scene['end_index']+1]
    tf=carla.Transform(carla.Location(*map(float,xyz)),carla.Rotation(yaw=float(-np.degrees(route[0,2]))));tf.location.z+=.5
    session.world.set_weather(carla.WeatherParameters.ClearNoon)
    bp=session.world.get_blueprint_library().find(c['vehicle'])
    if bp.has_attribute('color'):bp.set_attribute('color',bp.get_attribute('color').recommended_values[0])
    ego=session.spawn(bp,tf)
    return ego,route,world_route


class Rig:
    def __init__(self,session,ego,c,world_route,sequence):
        import carla
        self.session=session;self.ego=ego;self.c=c;self.sequence=sequence
        self.camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
        self.events=[];self.sync=FrameSynchronizer(CAMERAS,cadence_ticks=1,max_frames=8,timeout_ticks=5,on_event=self.events.append)
        self.imu=Queue(40);self.pending_imu={};self.pending={};self.states={};self.arrivals={};self.overflow=0
        self.collision=Queue(100);self.lane=Queue(100);self.collisions=[];self.lanes=[]
        self.drop_camera=None;self.corrupt_timestamp=False;self.arrival_records=[]
        # Every rig owns a distinct generation and inbox. Closing it precedes
        # sensor.stop/destroy, so in-flight old callbacks cannot feed a new rig.
        import uuid
        self.raw_inbox=RawInbox(sequence,session.world.id,uuid.uuid4().hex)
        self.raw_records=[];self.raw_seen={};self.raw_fault=None;self.sensor_ids={}
        self.snapshot_records=[]
        session.cleanup_callbacks.append(self.raw_inbox.close)
        def enqueue(q,value):
            try:q.put_nowait(value)
            except Full:self.overflow+=1
        for name in CAMERAS:
            s=self.camera['sensors'][name];bp=session.world.get_blueprint_library().find('sensor.camera.rgb')
            # Fixed accepted configuration: every physics step. This is not a
            # remedy for asynchronous server metadata/pixel association defects.
            for key,val in [('image_size_x',704),('image_size_y',396),('fov',s['fov']),('sensor_tick',0.)]:bp.set_attribute(key,str(val))
            cam=session.spawn(bp,carla.Transform(carla.Location(x=s['x'],y=s['y'],z=s['z']),carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll'])),parent=ego)
            self.sensor_ids[name]=cam.id
            cam.listen(self.raw_inbox.callback(name,cam.id))
        bp=session.world.get_blueprint_library().find('sensor.other.imu');bp.set_attribute('sensor_tick','0.0')
        session.spawn(bp,carla.Transform(carla.Location(x=-1.4)),parent=ego).listen(lambda x:enqueue(self.imu,x))
        session.spawn(session.world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego).listen(lambda x:enqueue(self.collision,x))
        session.spawn(session.world.get_blueprint_library().find('sensor.other.lane_invasion'),carla.Transform(),parent=ego).listen(lambda x:enqueue(self.lane,x))
        source=Path(c['runtime_root'])/'official-source/leaderboard/team_code/planner.py'
        spec=importlib.util.spec_from_file_location('j13_native_navigation',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.nav=module.RoutePlanner(4.,50.)
        from simulator.m8j13_navigation import set_navigation_route
        set_navigation_route(self.nav,[(SimpleNamespace(location=SimpleNamespace(x=float(p[0]),y=float(p[1]))),SimpleNamespace(value=4)) for p in world_route],
            coordinate_type='world_carla',client_version=session.client.get_client_version())
        self.last_frame=None

    def tick(self):
        start=time.monotonic();frame=self.session.world.tick(self.c['client_timeout_s']);end=time.monotonic()
        snap=self.session.world.get_snapshot()
        if snap.frame!=frame or (self.last_frame is not None and frame!=self.last_frame+1):raise RuntimeError('CompetingTickOwnerOrFrameGap')
        self.last_frame=frame;m=measurement(snap,self.ego)
        self.snapshot_records.append(dict(run=self.sequence,episode=self.session.world.id,frame=frame,
            elapsed_seconds=snap.timestamp.elapsed_seconds,delta_seconds=snap.timestamp.delta_seconds,
            ego=m,tick_start_monotonic_s=start,tick_return_monotonic_s=end))
        if self.sync.latest==-1:self.sync.begin_interval(frame,explicit_cadence=True)
        target,command=self.nav.run_step(np.array(m['position'][:2]))
        row=dict(frame_id=frame,timestamp_s=m['sim_s'],sequence_id=self.sequence,ego_world_matrix=m['pose'],speed_mps=m['speed'],
            navigation_command=command.value,target_world_xy=target.tolist(),ego_state_source='Oracle Ego State',
            input_capture_wall_s=start,tick_return_wall_s=end,world_tick_ms=(end-start)*1000)
        self.states[frame]=row;self.sync.add_snapshot(frame,m['sim_s'],{},[])
        while not self.collision.empty():
            e=self.collision.get_nowait();self.collisions.append(dict(frame=e.frame,timestamp=e.timestamp,other_type=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z]))
        while not self.lane.empty():
            e=self.lane.get_nowait();self.lanes.append(dict(frame=e.frame,timestamp=e.timestamp,markings=[str(x.type) for x in e.crossed_lane_markings]))
        return m

    def collect(self,frame,timeout=None):
        deadline=time.monotonic()+(self.c['max_sensor_wait_wall_s'] if timeout is None else timeout)
        while time.monotonic()<deadline:
            self.drain_raw()
            while True:
                try:x=self.imu.get_nowait()
                except Empty:break
                self.pending_imu[x.frame]=x
            for bundle in self.sync.drain():self.pending[bundle.frame_id]=bundle
            if self.overflow or self.sync.stats['queue_overflow']:raise RuntimeError('SensorQueueOverflow')
            if frame in self.pending and frame in self.pending_imu:
                b=self.pending.pop(frame);imu=self.pending_imu.pop(frame);row=self.states.pop(frame)
                row.update(imu=[imu.accelerometer.x,imu.accelerometer.y,imu.accelerometer.z,imu.gyroscope.x,imu.gyroscope.y,imu.gyroscope.z,imu.compass],imu_frame=imu.frame,imu_timestamp_s=imu.timestamp,
                    camera_frames={k:s.frame_id for k,s in b.sensors.items()},camera_timestamps_s={k:s.timestamp_s for k,s in b.sensors.items()},
                    camera_world_matrices={k:s.T_carla_world_from_camera.tolist() for k,s in b.sensors.items()})
                if self.corrupt_timestamp:row['camera_timestamps_s']['CAM_FRONT']+=.05
                images={k:np.frombuffer(s.raw_bgra,np.uint8).reshape(396,704,4)[:,:,:3].copy() for k,s in b.sensors.items()}
                assert_observation(row,images)
                if imu.frame!=frame or abs(imu.timestamp-row['timestamp_s'])>1e-5:raise RuntimeError('IMUTimeMismatch')
                row['sensor_ready_wall_s']=time.monotonic()
                return row,images
            time.sleep(.001)
        raise TimeoutError('SensorFrameTimeout:'+str(frame))

    def discard_before(self,frame):
        for store in (self.pending,self.pending_imu,self.states):
            for key in list(store):
                if key<frame:store.pop(key,None)
        for store in (self.sync.pending,self.sync.snapshots):
            for key in list(store):
                if key<frame:store.pop(key,None)
        self.sync.closed_through=max(self.sync.closed_through,frame-1)

    def available(self):
        """Drain complete measured frames without stalling the next world tick."""
        self.drain_raw()
        while True:
            try:x=self.imu.get_nowait()
            except Empty:break
            self.pending_imu[x.frame]=x
        for b in self.sync.drain():self.pending[b.frame_id]=b
        if self.overflow or self.sync.stats['queue_overflow']:raise RuntimeError('SensorQueueOverflow')
        ready=[]
        for frame in sorted(self.states):
            if frame not in self.pending or frame not in self.pending_imu:break
            b=self.pending.pop(frame);imu=self.pending_imu.pop(frame);row=self.states.pop(frame)
            row.update(imu=[imu.accelerometer.x,imu.accelerometer.y,imu.accelerometer.z,imu.gyroscope.x,imu.gyroscope.y,imu.gyroscope.z,imu.compass],imu_frame=imu.frame,imu_timestamp_s=imu.timestamp,
                camera_frames={k:s.frame_id for k,s in b.sensors.items()},camera_timestamps_s={k:s.timestamp_s for k,s in b.sensors.items()},
                camera_world_matrices={k:s.T_carla_world_from_camera.tolist() for k,s in b.sensors.items()})
            if self.corrupt_timestamp:row['camera_timestamps_s']['CAM_FRONT']+=.05
            images={k:np.frombuffer(s.raw_bgra,np.uint8).reshape(396,704,4)[:,:,:3].copy() for k,s in b.sensors.items()}
            assert_observation(row,images)
            if imu.frame!=frame or abs(imu.timestamp-row['timestamp_s'])>1e-5:raise ValueError('IMUTimeMismatch')
            row['sensor_ready_wall_s']=time.monotonic();ready.append((row,images))
        return ready

    def drain_raw(self):
        """Preserve raw entry values and fail closed on duplicate/changed fields.

        A late duplicate may follow an already released group. Latch the fault
        instead of treating it as a harmless duplicate or rewriting provenance.
        This detects the problem; it cannot retroactively validate that group.
        """
        for stamp,image in self.raw_inbox.drain():
            record=stamp.record();self.raw_records.append(record)
            self.arrival_records.append((stamp.camera,stamp.frame,stamp.timestamp_s,stamp.receive_monotonic_s))
            key=(stamp.camera,stamp.frame)
            if key in self.raw_seen:
                self.raw_fault='DuplicateRawSensorFrame'
                self.events.append(dict(reason=self.raw_fault,frame_id=stamp.frame,sensor_id=stamp.camera,
                    previous_timestamp_s=self.raw_seen[key],timestamp_s=stamp.timestamp_s))
            self.raw_seen[key]=stamp.timestamp_s
            if image.frame!=stamp.frame or image.timestamp!=stamp.timestamp_s:
                self.raw_fault='RawImageFieldsChanged'
            if stamp.actor_id!=self.sensor_ids[stamp.camera]:self.raw_fault='WrongSensorInstance'
            if stamp.camera!=self.drop_camera:self.sync.callback(stamp.camera)(image)
        if self.raw_inbox.stats['queue_overflow']:self.raw_fault='RawSensorQueueOverflow'
        if self.raw_fault:raise ValueError(self.raw_fault)

    def overdue(self,sim_s,wall_s):
        if not self.states:return None
        row=self.states[min(self.states)]
        if sim_s-row['timestamp_s']>self.c['max_plan_age_sim_s']+1e-6 or wall_s-row['input_capture_wall_s']>self.c['max_plan_age_wall_s']+1e-6:
            return dict(frame=row['frame_id'],sim_age_s=sim_s-row['timestamp_s'],wall_age_s=wall_s-row['input_capture_wall_s'])
        return None


def route_metrics(route,m):
    pos=np.array(m['position'][:2])*[1,-1];segments=np.diff(route[:,:2],axis=0);lens=np.linalg.norm(segments,axis=1)
    alpha=np.clip(np.sum((pos-route[:-1,:2])*segments,axis=1)/np.maximum(lens**2,1e-8),0,1)
    closest=route[:-1,:2]+alpha[:,None]*segments;dist=np.linalg.norm(closest-pos,axis=1);i=int(dist.argmin())
    arc=np.r_[0,np.cumsum(lens)];progress=arc[i]+alpha[i]*lens[i]
    heading=(float(-m['yaw']-route[i,2])+np.pi)%(2*np.pi)-np.pi
    return dict(route_completion=float(progress/arc[-1]),lateral_error_m=float(dist[i]),heading_error_rad=float(heading),endpoint_distance_m=float(np.linalg.norm(pos-route[-1,:2])))
