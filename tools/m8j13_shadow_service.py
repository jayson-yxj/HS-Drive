"""Frozen FP32 shadow worker: no CARLA client, actor, or executable control IPC."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from tools.m8j12_runtime import build,reset,state_digest,observation_batch
from tools.m8j12_replay import finite,plain,memory,install_selection_log
from datasets.m8j12_contract import official_camera_contract,planning_to_hs,DryRunController,measured_navigation_target,assert_observation
from simulator.m8j13_safety import PlanStamp,freshness,validate_trajectory
from tools.m8j13_common import config,atomic,write,append,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    os.umask(0o077);c=config();cfg_run=json.loads(Path('configs/m8j13_compatibility.json').read_text())
    ledger=root.parent/'forward_budget.jsonl';results=[];rejected=0;model=None;baseline=None
    allowed={'frame_id','timestamp_s','sequence_id','ego_world_matrix','speed_mps','imu','imu_frame','imu_timestamp_s',
        'navigation_command','target_world_xy','ego_state_source','input_capture_wall_s','camera_frames','camera_timestamps_s','camera_world_matrices'}
    def age_reason(row,start):
        state=json.loads((root/'current_snapshot.json').read_text());now=time.monotonic()
        stamp=PlanStamp(row['sequence_id'],row['frame_id'],row['timestamp_s'],row['input_capture_wall_s'],start,now,'frozen')
        return freshness(stamp,row['sequence_id'],state['frame'],state['sim_s'],now)
    try:
        torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model,model_cfg,load=build();model.cuda();reset(model);baseline=state_digest(model)
        versions={n:p._version for n,p in model.named_parameters()}
        camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
        ctrl=DryRunController(Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py')
        selection={};install_selection_log(model,selection)
        atomic(root/'model_ready.json',dict(load=load,torch=torch.__version__,cuda=torch.version.cuda,
            parameters_frozen=all(not p.requires_grad for p in model.parameters()),eval=not model.training,
            fp32=all(p.dtype==torch.float32 for p in model.parameters()),model_controls=0,network_namespace_isolated=True,
            pid_source_sha256=digest(Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py')))
        seen=set();previous=None
        while not (root/'model_stop').exists():
            requests=sorted((root/'requests').glob('*.json')) if (root/'requests').exists() else []
            for request in requests:
                if request.name in seen:continue
                seen.add(request.name);row=json.loads(request.read_text());start=time.monotonic()
                if set(row)!=allowed:raise RuntimeError('UnexpectedShadowInputKeys')
                images=dict(np.load(request.with_suffix('.npz'),allow_pickle=False));assert_observation(row,images)
                if row['imu_frame']!=row['frame_id'] or abs(row['imu_timestamp_s']-row['timestamp_s'])>1e-5:raise RuntimeError('IMUContract')
                reason=age_reason(row,start)
                if reason:
                    rejected+=1;atomic(root/'responses'/request.name,dict(status='InputRejected',reason=reason,model_forward=False));continue
                batch=observation_batch(model_cfg,row,images,camera);torch.cuda.synchronize();prepared=time.monotonic()
                reason=age_reason(row,start)
                if reason:
                    rejected+=1;atomic(root/'responses'/request.name,dict(status='InputRejectedAfterPreprocess',reason=reason,model_forward=False));continue
                if previous is not None and row['timestamp_s']<=previous:raise RuntimeError('NonIncreasingModelTimestamp')
                before=memory(model)
                if before['history_timestamp'] is not None and any(t>=row['timestamp_s'] for t in before['history_timestamp']):raise RuntimeError('FutureModelMemory')
                with (root.parent/'forward_budget.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    count=len(ledger.read_text().splitlines()) if ledger.exists() else 0
                    if count>=cfg_run['max_forward_attempts'] or len(results)>=cfg_run['planned_forwards']:raise RuntimeError('ForwardBudgetExhausted')
                    append(ledger,dict(attempt=count+1,kind='live_shadow',request=request.name,wall_s=time.monotonic()))
                selection.clear();torch.cuda.reset_peak_memory_stats();infer=time.monotonic()
                with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
                torch.cuda.synchronize();inferred=time.monotonic()
                if not finite(prediction):raise RuntimeError('NonfiniteNativeOutput')
                trajectory,meta=planning_to_hs(prediction,row['timestamp_s']);converted=time.monotonic()
                control=ctrl.command(prediction,row['speed_mps'],measured_navigation_target(row));finished=time.monotonic()
                reason,diagnostics=validate_trajectory(trajectory.points,row['speed_mps'],c['trajectory_checks'])
                if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('ParameterChanged')
                record=dict(status='ShadowDiagnostic',source='FrozenModelDryRun',frame=row['frame_id'],timestamp_s=row['timestamp_s'],
                    input_wall_s=row['input_capture_wall_s'],native_path=plain(prediction['lat_reg_final']),native_speeds=plain(prediction['lon_reg_final']),
                    native_trajectory=plain(prediction['traj_final']),hs_points=trajectory.points.tolist(),hs_metadata=meta,
                    raw_pid=plain(control['raw_pid']),native_agent_control=control['native_agent_control'],applied_to_vehicle=False,
                    trajectory_check_reason=reason,trajectory_diagnostics=diagnostics,result_age_reason=age_reason(row,start),
                    memory_before=before,memory_after=memory(model),timestamp_gap_s=None if previous is None else row['timestamp_s']-previous,
                    timings_ms=dict(read_and_preprocess=(prepared-start)*1000,inference=(inferred-infer)*1000,
                        mapping=(converted-inferred)*1000,pid=(finished-converted)*1000,total=(finished-start)*1000),
                    gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved(),finite=True)
                torch.save(dict(prediction=prediction,selection=selection),root/'native'/(request.stem+'.pt'))
                write(root/'diagnostics'/request.name,record);results.append(record);previous=row['timestamp_s']
                # Owner receives completion metadata only, never steer/throttle.
                atomic(root/'responses'/request.name,dict(status='ShadowComplete',model_forward=True,finite=True,result_expired=record['result_age_reason'] is not None))
                if len(results)==cfg_run['planned_forwards']:break
            if len(results)==cfg_run['planned_forwards']:break
            time.sleep(.002)
    except BaseException as exc:atomic(root/'model_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    finally:
        unchanged=None
        if model is not None and baseline is not None:reset(model);unchanged=state_digest(model)==baseline
        atomic(root/'model_final.json',dict(completed_forwards=len(results),input_rejections=rejected,
            fixed_state_unchanged=unchanged,model_controls=0,optimizer_steps=0,
            full_forward_attempts=len(ledger.read_text().splitlines()) if ledger.exists() else 0))


if __name__=='__main__':main()
