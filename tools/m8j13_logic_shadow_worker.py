"""One frozen model load, serial logical shadow tasks, diagnostic-only outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from tools.m8j12_runtime import build,reset,state_digest
from tools.m8j12_replay import memory,plain,finite
from tools.m8j13_preprocess import OfficialPreprocessor
from tools.m8j13_logic_shadow_diagnostics import transported_pid
from tools.m8j13_common import config,atomic,write,append
from tools.m8j13_logic_shutdown import final_model_record
from datasets.m8j12_contract import CAMERAS,assert_observation,official_camera_contract,planning_to_hs,DryRunController
from simulator.m8j13_logic_shadow_budget import validate_shadow_budget
from simulator.m8j13_safety import validate_trajectory
from simulator.m8j13_logic_shadow import PROTOCOL,CHECKS,task_valid,require_notification,memory_valid


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;os.umask(0o077)
    cfg=json.loads((root/'frozen_config.json').read_text());c=config();model=None;baseline=None;attempts=success=pid_calls=completed=0;previous=None;identity=None;last_frame=None
    ledger=root/'forward_attempts.jsonl';pid_ledger=root/'pid_attempts.jsonl'
    try:
        validate_shadow_budget(cfg)
        atomic(root/'model_progress.json',dict(stage='loading',attempts=0,successful_forwards=0,pid_calls=0,wall_s=time.monotonic()))
        if cfg['shadow_protocol']!=PROTOCOL:raise ValueError('ExplicitLogicalShadowOnly')
        loading=time.monotonic();torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model,mc,loaded=build();model.cuda();reset(model);baseline=state_digest(model)
        versions={n:p._version for n,p in model.named_parameters()}
        official=Path(c['runtime_root'])/'official-source/leaderboard/team_code'
        camera=official_camera_contract(official/'sparsedrive_b2d_agent.py');pre=OfficialPreprocessor(mc,camera);ctrl=DryRunController(official/'pid_controller.py')
        if 'carla' in sys.modules:raise ValueError('ForbiddenCARLAInModel')
        atomic(root/'model_ready.json',dict(load=loaded,load_wall_s=time.monotonic()-loading,initial_memory=memory(model),parameters_frozen=all(not p.requires_grad for p in model.parameters()),eval=not model.training,fp32=all(p.dtype==torch.float32 for p in model.parameters()),network_namespace=os.readlink('/proc/self/ns/net'),gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),model_controls=0))
        seen=set()
        while not (root/'model_stop').exists():
            requests=sorted((root/'shadow_tasks').glob('*.json'))
            path=next((p for p in requests if p.name not in seen),None)
            if path is None:time.sleep(.002);continue
            prediction=None;control=None;transport=None;trajectory=None;times={}
            if (root/'model_stop').exists():break
            received=time.monotonic();packet=json.loads(path.read_text());seen.add(path.name)
            if set(packet)!={'task','row'}:raise ValueError('ShadowTaskPayloadKeys')
            t=packet['task'];row=packet['row'];times=dict(worker_received=received)
            if identity is None:identity=t['identity']
            if identity!=t['identity'] or row['sequence_id']!=identity[0]:raise ValueError('ShadowInputIdentity')
            if t['sequence']!=completed+1 or t['sequence']>24 or t['frame']!=row['frame_id'] or t['input_sim_s']!=row['timestamp_s'] or t['input_wall_s']!=row['input_capture_wall_s']:raise ValueError('ShadowInputMismatch')
            if previous is not None and (row['frame_id']!=last_frame+4 or abs(row['timestamp_s']-previous-.2)>1e-5):raise ValueError('ShadowMemoryCadence')
            allowed={'frame_id','timestamp_s','sequence_id','ego_world_matrix','speed_mps','imu','imu_frame','imu_timestamp_s','navigation_command','target_world_xy','ego_state_source','input_capture_wall_s','camera_frames','camera_timestamps_s','camera_world_matrices'}
            if set(row)!=allowed or row['ego_state_source']!='Oracle Ego State':raise ValueError('UnexpectedShadowInputBoundary')
            if row['imu_frame']!=row['frame_id'] or abs(row['imu_timestamp_s']-row['timestamp_s'])>1e-5:raise ValueError('ShadowIMUIdentity')
            folder=root/'inputs'/str(t['frame'])
            images={n:np.frombuffer((folder/(n+'.bgra')).read_bytes(),np.uint8).reshape(396,704,4)[:,:,:3].copy() for n in CAMERAS}
            times['read_done']=time.monotonic();assert_observation(row,images);task_valid(t,time.monotonic(),t['frozen'])
            times['preprocess_start']=time.monotonic();cpu,pre_timing=pre.cpu(row,images);times['preprocess_done']=time.monotonic()
            batch,transfer_ms=pre.gpu(cpu);times['transfer_done']=time.monotonic();task_valid(t,time.monotonic(),t['frozen'])
            before=memory(model)
            if previous is None and (before['cached'] or before['history_timestamp'] is not None):raise ValueError('MemoryNotReset')
            if before['history_timestamp'] is not None and any(x>=row['timestamp_s'] for x in before['history_timestamp']):raise ValueError('FutureMemory')
            if attempts>=24:raise ValueError('ForwardBudgetExhausted')
            if (root/'model_stop').exists():break
            task_valid(t,time.monotonic(),t['frozen']);attempts+=1
            append(ledger,dict(attempt=attempts,sequence=t['sequence'],frame=t['frame'],timestamp=row['timestamp_s'],wall_s=time.monotonic()))
            atomic(root/'model_progress.json',dict(stage='forward_in_flight',attempts=attempts,successful_forwards=success,pid_calls=pid_calls,task_sequence=t['sequence'],wall_s=time.monotonic()))
            torch.cuda.reset_peak_memory_stats();times['forward_start']=time.monotonic()
            with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
            torch.cuda.synchronize();times['forward_done']=time.monotonic();success+=1
            if (root/'model_stop').exists():break
            validation_start=times['forward_done'];validation_deadline=min(validation_start+.1,t['deadline_wall_s'],t['input_wall_s']+4.,t['run_deadline'])
            if not finite(prediction):raise ValueError('NonfiniteNativeOutput')
            trajectory,meta=planning_to_hs(prediction,row['timestamp_s']);times['hs_done']=time.monotonic()
            reason,diagnostics=validate_trajectory(trajectory.points,row['speed_mps'],c['trajectory_checks'])
            if reason:raise ValueError(reason)
            invoke,transport=transported_pid(prediction,row,t['frozen'],ctrl);times['transport_done']=time.monotonic()
            if pid_calls+cfg.get('preflight_pid_calls',0)>=cfg['max_pid_calls']:raise ValueError('PIDBudgetExhausted')
            append(pid_ledger,dict(sequence=t['sequence'],frame=t['frame'],wall_s=time.monotonic()));pid_calls+=1
            control,raw_error,agent_error=invoke();times['pid_done']=time.monotonic()
            if raw_error or agent_error:raise ValueError(raw_error or agent_error)
            if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('ParameterChanged')
            after=memory(model);memory_valid(before,after,previous,row['timestamp_s'])
            task_valid(t,time.monotonic(),t['frozen']);times['checks_done']=time.monotonic()
            if times['checks_done']>validation_deadline:raise ValueError('ShadowValidationDeadline')
            times['publish_start']=time.monotonic()
            reply=dict(protocol=PROTOCOL,identity=t['identity'],sequence=t['sequence'],frame=t['frame'],input_sim_s=t['input_sim_s'],input_wall_s=t['input_wall_s'],task_deadline=t['deadline_wall_s'],validation_start=validation_start,validation_deadline=validation_deadline,checks={k:True for k in CHECKS},times=times,model_forward=True)
            require_notification(reply);atomic(root/'shadow_completions'/path.name,reply);published=time.monotonic()
            record=dict(task=t,times=times,published_wall_s=published,preprocessing=pre_timing,gpu_transfer_ms=transfer_ms,memory_before=before,memory_after=after,timestamp_gap=None if previous is None else row['timestamp_s']-previous,
                native_path=plain(prediction['lat_reg_final']),native_speeds=plain(prediction['lon_reg_final']),native_trajectory=plain(prediction['traj_final']),hs_points=trajectory.points.tolist(),hs_timestamp=trajectory.timestamp_s,hs_metadata=meta,
                transport=transport,raw_pid=plain(control['raw_pid']),native_agent_control=control['native_agent_control'],pid_metadata=plain(control['pid_metadata']),trajectory_check_reason=reason,trajectory_diagnostics=diagnostics,
                raw_control_reason=raw_error,agent_control_reason=agent_error,model_controls=0,gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved())
            save=time.monotonic();torch.save(prediction,root/'native'/('%06d.pt'%t['sequence']));saved=time.monotonic()
            record['native_save_ms']=(saved-save)*1000;write(root/'diagnostics'/path.name,record);logged=time.monotonic()
            append(root/'worker_timing.jsonl',dict(sequence=t['sequence'],published_wall_s=published,archive_end_wall_s=logged,archive_ms=(logged-save)*1000,json_write_ms=(logged-saved)*1000))
            previous=row['timestamp_s'];last_frame=row['frame_id'];completed+=1
            del batch,cpu,prediction,images
    except BaseException as exc:
        error=dict(error=str(exc),traceback=traceback.format_exc(),attempts=attempts,successful_forwards=success,pid_calls=pid_calls,wall_s=time.monotonic())
        atomic(root/'model_error.json',error)
        # Preserve failed output evidence too; never publish it as a valid completion.
        try:
            if locals().get('prediction') is not None:torch.save(prediction,root/'native'/('failed_%06d.pt'%attempts))
            atomic(root/'failed_diagnostic.json',dict(error=error,times=locals().get('times'),task=locals().get('t'),control=plain(locals().get('control')),transport=locals().get('transport'),memory=None if model is None else memory(model)))
        except Exception as archive_error:atomic(root/'archive_error.json',dict(error=repr(archive_error)))
    finally:
        final_model_record(root,dict(attempts=attempts,successful_forwards=success,completed=completed,pid_calls=pid_calls),model,baseline,state_digest,memory)

if __name__=='__main__':main()
