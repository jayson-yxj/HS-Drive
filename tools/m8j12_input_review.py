"""Frozen-source input boundary and camera geometry review, no CARLA launch."""
from pathlib import Path
import json,hashlib
import numpy as np
from datasets.m8j12_contract import official_camera_contract,CAMERAS

def main():
    c=json.loads(Path('configs/m8j12_protocol.json').read_text());src=Path(c['runtime_root'])/'official-source'
    contract=official_camera_contract(src/'leaderboard/team_code/sparsedrive_b2d_agent.py')
    data={'official_commit':c['official_commit'],'source_evidence':{str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [src/'leaderboard/team_code/sparsedrive_b2d_agent.py',src/'projects/configs/sparsedrive_stage2.py',src/'leaderboard/team_code/pid_controller.py']},
      'camera_order':CAMERAS,'sensors':contract['sensors'],'capture_wh':[704,396],'network_wh':[704,384],'crop':[0,12,704,396],'JPEG_quality':20,'normalization':{'mean':[123.675,116.28,103.53],'std':[58.395,57.12,57.375],'to_rgb':True},
      'model_inputs':['six_camera_RGB','timestamp_measured_s','projection_mat','image_wh','ego_status','gt_ego_fut_cmd_navigation_not_future_trajectory','tp_near','img_metas_T_global_T_global_inv_timestamp'],
      'allowed_oracle_inputs':['current_ego_pose','speed','IMU_acceleration_angular_velocity_heading','fixed_navigation_route'],
      'excluded':['LiDAR_payload','Radar','ActorBoxes','mapGT','traffic_target_states','future_executed_trajectory','topdown_image'],
      'source_findings':[
        'Agent sensors subscribes LiDAR and BEV. tick queries a LiDAR Actor transform for pose; points are not read by model.',
        'Offline mapping computes same fixed virtual-sensor pose from recorded ego pose; no fake pointcloud and no world query in replay.',
        'BEV is used for visualization, not Collect model input. Offline path does not request BEV.',
        'Agent self.step/frame_rate is replaced by measured simulation timestamp, never relabel file counters.',
        'gt_ego_fut_cmd name is route command, not a future GT trajectory. cond_config actually enables target_point only.',
        'ego_status tail is zero reserved slot; no historical model or collection control enters active condition encoder.',
        'get_metric_info queries benchmark criteria after control and is omitted from standalone replay.',
        'Native planning coordinates are ego-origin x right/y forward (dataset get_trajs mode ego); detection references virtual LiDAR origin.'
      ],'timing_s':{'physics':.1,'camera':.1,'inference':.1,'planning':.5},'input_boundary_status':'SourceReviewed_RealCaptureNotRun','controls':'Original PID numeric output only; no CARLA handle in inference/controller process'}
    # Use CARLA SDK only as offline transform math, never Client/world.
    import carla
    O=np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],float)
    native_to_carla=np.array([[0,1,0,-.39],[1,0,0,0],[0,0,1,1.84],[0,0,0,1]],float)
    errors={}
    for name,s in contract['sensors'].items():
        t=carla.Transform(carla.Location(s['x'],s['y'],s['z']),carla.Rotation(pitch=s['pitch'],yaw=s['yaw'],roll=s['roll']))
        computed=O@np.linalg.inv(np.array(t.get_matrix()))@native_to_carla
        error=float(abs(computed-contract['lidar2cam'][name].T).max());assert error<2e-6,(name,error);errors[name]=error
    data['calibration_math_test_max_abs']=errors
    Path('reports/m8j12_sources/input_boundary.json').write_text(json.dumps(data,indent=2)+'\n');print('Official camera transforms verified with CARLA SDK math only',errors)
if __name__=='__main__':main()
