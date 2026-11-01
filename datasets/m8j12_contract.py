"""SparseDriveV2 B2D-only observation/trajectory boundary. No vehicle actuation."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
from simulator.oracle_driver import Trajectory

CAMERAS=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')
MODEL_KEYS={'img','timestamp','projection_mat','image_wh','ego_status','gt_ego_fut_cmd','tp_near','img_metas'}
LEFT_TO_RIGHT=np.diag([1.,-1.,1.,1.])
CARLA_TO_NATIVE=np.array([[0.,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])


def official_camera_contract(source):
    """Read only the frozen Agent's sensor method and numeric calibration assignments.

    Does not import/instantiate its world-dependent Agent or subscribe LiDAR/BEV.
    """
    tree=ast.parse(Path(source).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SparseDriveAgent')
    sensors=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sensors')
    scope={'resize_scale':.44,'IS_BENCH2DRIVE':False}
    exec(compile(ast.Module(body=[sensors],type_ignores=[]),str(source),'exec'),scope)
    rgb=[s for s in scope['sensors'](None) if s['type']=='sensor.camera.rgb']
    obj=SimpleNamespace();scope={'np':np,'self':obj}
    setup=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='setup')
    for name in ('lidar2cam','cam_intrinsic','lidar2ego'):
        assign=next(n for n in setup.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr==name)
        exec(compile(ast.Module(body=[assign],type_ignores=[]),str(source),'exec'),scope)
    scale=np.diag([.44,.44,1.])
    K={c:scale@obj.cam_intrinsic[c] for c in CAMERAS}
    P={}
    for c in CAMERAS:
        view=np.eye(4);view[:3,:3]=K[c];P[c]=view@obj.lidar2cam[c].T
    return dict(sensors={s['id']:s for s in rgb},K=K,projection=P,lidar2cam=obj.lidar2cam,lidar2ego=obj.lidar2ego,
                capture_wh=(704,396),network_wh=(704,384),crop=(0,12,704,396),jpeg_quality=20)


def native_global_pose(carla_world_from_ego):
    """Oracle Ego State + fixed virtual sensor extrinsic; no LiDAR Actor or payload."""
    pose=np.asarray(carla_world_from_ego,dtype=np.float64)
    assert pose.shape==(4,4) and np.isfinite(pose).all()
    ego_from_virtual=np.eye(4);ego_from_virtual[:3,3]=[-.39,0,1.84]
    world_from_virtual=pose@ego_from_virtual
    return np.linalg.inv(CARLA_TO_NATIVE@np.linalg.inv(world_from_virtual)@LEFT_TO_RIGHT)


def native_target(carla_world_from_ego,target_world_xy):
    pose=np.asarray(carla_world_from_ego,float);point=np.r_[target_world_xy,pose[2,3],1.]
    ego=np.linalg.inv(pose)@point
    return ego[[1,0]].astype(np.float32) # right, forward; ego origin, not virtual LiDAR origin


def measured_navigation_target(row):
    """Official live Agent's planar compass transform, including measured heading."""
    pose=np.asarray(row['ego_world_matrix']);delta=np.asarray(row['target_world_xy'])-pose[:2,3]
    theta=float(row['imu'][-1]);c,s=np.cos(theta),np.sin(theta)
    return (np.array([[c,-s],[s,c]])@(delta*[1,-1])).astype(np.float32)


def assert_observation(row,images):
    required={'frame_id','timestamp_s','sequence_id','ego_world_matrix','speed_mps','imu','navigation_command','target_world_xy','camera_frames','camera_timestamps_s'}
    if not required.issubset(row):raise ValueError('MissingObservationFields')
    if type(row['frame_id']) is not int or not np.isfinite(row['timestamp_s']):raise ValueError('InvalidMeasuredTime')
    if set(images)!=set(CAMERAS):raise ValueError('CameraSetMismatch')
    for c in CAMERAS:
        if images[c].shape!=(396,704,3) or images[c].dtype!=np.uint8:raise ValueError('CameraContractMismatch')
        if row['camera_frames'][c]!=row['frame_id'] or abs(row['camera_timestamps_s'][c]-row['timestamp_s'])>1e-5:raise ValueError('UnsynchronizedMeasurement')
    if not 1<=row['navigation_command']<=6:raise ValueError('UnknownNavigationCommand')
    if np.asarray(row['imu']).shape!=(7,) or not np.isfinite(row['imu']).all():raise ValueError('InvalidIMU')
    if not np.isfinite(row['speed_mps']):raise ValueError('InvalidSpeed')


def planning_to_hs(output,timestamp_s):
    """Native ego right/forward -> HS forward/left, preserving six 0.5s points.

    Speed is official window-average lon output; yaw is derived segment heading,
    not a new model prediction. No tail extrapolation or safety score is added.
    """
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    native=array(output['traj_final']);velocity=array(output['lon_reg_final']);path=array(output['lat_reg_final'])
    if native.shape!=(6,2) or velocity.shape!=(6,) or path.shape!=(15,2):raise ValueError('PlanningShapeMismatch')
    if not all(np.isfinite(x).all() for x in (native,velocity,path)) or not np.isfinite(timestamp_s):raise ValueError('NonfinitePlanning')
    xy=np.stack([native[:,1],-native[:,0]],axis=-1)
    delta=np.diff(np.vstack([np.zeros((1,2)),xy]),axis=0)
    yaw=np.arctan2(delta[:,1],delta[:,0])
    trajectory=Trajectory(np.column_stack([xy,yaw,velocity]),float(timestamp_s),.5,'SparseDriveV2_prediction_OfflineDryRun')
    metadata=dict(start_offset_s=.5,horizon_s=3.,coordinate='hs_ego_x_forward_y_left_m',yaw_source='derived_segment_heading',
        speed_source='native_lon_window_average_mps',path_spacing_m=1.,path_not_time_sampled=True,model_control_applied=0)
    return trajectory,metadata


class DryRunController:
    """Original PID and Agent postprocessing, numeric return only, no actor handle."""
    def __init__(self,pid_source):
        spec=importlib.util.spec_from_file_location('m8j12_official_pid',str(pid_source));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.controller=module.PIDController(None)
    def command(self,output,speed,target):
        raw=self.controller.control_pid(output,np.float64(speed),np.asarray(target))
        steer,throttle,brake,metadata=raw
        if not np.isfinite([steer,throttle,brake]).all():raise ValueError('NonfiniteNativeControl')
        raw_values=dict(steer=float(steer),throttle=float(throttle),brake=float(brake))
        if brake<.05:brake=0.
        if throttle>brake:brake=0.
        final=dict(steer=float(np.clip(steer,-1,1)),throttle=float(np.clip(throttle,0,1)),brake=float(np.clip(brake,0,1)))
        return dict(raw_pid=raw_values,native_agent_control=final,pid_metadata=metadata,applied_to_vehicle=False)
