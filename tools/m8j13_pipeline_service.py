"""Opt-in two-lease shadow worker: one CPU preparation, serial model/PID.

Only completion flags cross to the owner. Required checks precede publication;
private input/prediction persistence follows publication and is still timed.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
from queue import Empty
import sys
import threading
import time
import traceback
import numpy as np
import torch
from tools.m8j12_runtime import build,reset,state_digest
from tools.m8j12_replay import finite,plain,memory,install_selection_log
from tools.m8j13_preprocess import OfficialPreprocessor
from tools.m8j13_common import config,atomic,write,append,digest
from datasets.m8j12_contract import official_camera_contract,planning_to_hs,DryRunController,measured_navigation_target,assert_observation
from simulator.m8j13_safety import PlanStamp,freshness,validate_trajectory
from simulator.m8j13_transport import Slots
from simulator.m8j13_prepared_slot import PreparedSlot

ALLOWED={'frame_id','timestamp_s','sequence_id','ego_world_matrix','speed_mps','imu','imu_frame','imu_timestamp_s',
    'navigation_command','target_world_xy','ego_state_source','input_capture_wall_s','camera_frames','camera_timestamps_s','camera_world_matrices'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-config',type=Path,required=True);a=p.parse_args()
    root=a.root;cfg=json.loads(a.run_config.read_text());c=config();os.umask(0o077)
    stop=threading.Event();pending=PreparedSlot();thread=None;model=None;baseline=None
    completed=0;rejected=0;attempt=None;previous=None;ledger=root.parent/'forward_budget.jsonl'
    def age(row,start,wait_for_snapshot=False):
        until=min(time.monotonic()+.06,row['input_capture_wall_s']+.3)
        while True:
            pending_path=root/'tick_pending.json'
            first=json.loads(pending_path.read_text()) if cfg.get('per_camera') and pending_path.exists() else None
            state=json.loads((root/'current_snapshot.json').read_text());now=time.monotonic()
            second=json.loads(pending_path.read_text()) if first is not None else None
            confirmed=first is None or (first==second and not second['pending'] and state['frame']==second['frame'])
            stamp=PlanStamp(row['sequence_id'],row['frame_id'],row['timestamp_s'],row['input_capture_wall_s'],start,now,'frozen')
            reason=freshness(stamp,row['sequence_id'],state['frame'],state['sim_s'],now)
            if not confirmed and not reason:reason='SimulationSnapshotUnconfirmed'
            if reason=='SimulationSnapshotUnconfirmed' and wait_for_snapshot and now<until:time.sleep(.001);continue
            return dict(reason=reason,wall_age_s=now-row['input_capture_wall_s'],sim_age_s=state['sim_s']-row['timestamp_s'],
                checked_wall_s=now,snapshot_confirmed=confirmed,sim_age_scope='Latest confirmed owner snapshot; unconfirmed in-flight tick never qualifies as fresh')
    def respond(name,status,**flags):atomic(root/'responses'/name,dict(status=status,**flags))
    def prepare_loop(model_cfg,camera):
        # This thread owns this stateless official Compose instance exclusively.
        prep=OfficialPreprocessor(model_cfg,camera);slots=Slots(json.loads((root/'transport.json').read_text())['root'])
        seen=set();identity=None
        try:
            while not stop.is_set() and not (root/'model_stop').exists():
                if not pending.reserve():continue
                paths=[x for x in sorted((root/'requests').glob('*.json')) if x.name not in seen]
                if not paths:pending.cancel();stop.wait(.002);continue
                request=paths[0];seen.add(request.name);start=time.monotonic();cpu_start=time.thread_time()
                row,images,actual=slots.read(json.loads(request.read_text()));read_done=time.monotonic()
                if set(row)!=ALLOWED:raise RuntimeError('UnexpectedShadowInputKeys')
                if identity is None:identity=actual
                if actual!=identity or actual['run']!=row['sequence_id']:raise RuntimeError('CrossGenerationInput')
                if set(actual['sensors'])!=set(images):raise RuntimeError('SensorIdentityMismatch')
                assert_observation(row,images)
                if row['imu_frame']!=row['frame_id'] or abs(row['imu_timestamp_s']-row['timestamp_s'])>1e-5:raise RuntimeError('IMUContract')
                write(root/'input_rows'/request.name,row)
                item=dict(name=request.name,row=row,images=images,start=start,read_ms=(read_done-start)*1000)
                checked=age(row,start);item['checked']=checked
                if checked['reason']:item['rejection']='InputRejected'
                else:
                    pre=time.monotonic();item['cpu_batch'],item['preprocessing']=prep.cpu(row,images)
                    item['preprocess_start']=pre;item['prepared']=time.monotonic();item['checked']=age(row,start)
                    if item['checked']['reason']:item['rejection']='InputRejectedAfterCPUPreprocess'
                item['prepare_thread_cpu_ms']=(time.thread_time()-cpu_start)*1000
                pending.publish(item)  # permit prevents a second queued/preparing item
        except BaseException as exc:pending.publish(dict(error=repr(exc),traceback=traceback.format_exc()))
    try:
        loading=time.monotonic();torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model,model_cfg,load=build();model.cuda();reset(model);baseline=state_digest(model);versions={n:p._version for n,p in model.named_parameters()}
        camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
        ctrl=DryRunController(Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py')
        selection={};install_selection_log(model,selection)
        if 'carla' in sys.modules:raise RuntimeError('ForbiddenSimulatorModuleInModelProcess')
        atomic(root/'model_ready.json',dict(load=load,parameters_frozen=all(not p.requires_grad for p in model.parameters()),eval=not model.training,
            fp32=all(p.dtype==torch.float32 for p in model.parameters()),torch=torch.__version__,cuda=torch.version.cuda,model_controls=0,
            network_namespace=os.readlink('/proc/self/ns/net'),load_and_digest_wall_s=time.monotonic()-loading,initial_memory=memory(model),
            pid_source_sha256=digest(Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py'),
            threads=dict(torch=torch.get_num_threads(),interop=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)))))
        if cfg.get('per_camera'):
            from tools.m8j13_camera_live import camera_prepare_loop
            target=lambda:camera_prepare_loop(root,model_cfg,camera,pending,stop,age)
        else:target=lambda:prepare_loop(model_cfg,camera)
        thread=threading.Thread(target=target,name='single-official-preprocessor',daemon=True);thread.start()
        while not (root/'model_stop').exists() or not pending.empty():
            try:item=pending.take(timeout=.02)
            except Empty:continue
            if 'error' in item:raise RuntimeError(item)
            if item.get('cancelled'):
                rejected+=1;append(root/'worker_timing.jsonl',dict(request=item['name'],status='PartialCancelled',start_wall_s=item['start']))
                respond(item['name'],'PartialCancelled',model_forward=False,reason='OwnerInputRejected');continue
            attempt=None;row=item['row'];start=item['start'];name=item['name'];wait_done=time.monotonic()
            checked=age(row,start,wait_for_snapshot=True);rejection=item.get('rejection')
            if not rejection and checked['reason']:rejection='InputRejectedAfterWait'
            if completed>=cfg['planned_forwards']:rejection='ForwardBudgetEnded'
            transfer=0.
            if not rejection:
                batch,transfer=OfficialPreprocessor.gpu(item.pop('cpu_batch'));checked=age(row,start,wait_for_snapshot=True)
                if checked['reason']:rejection='InputRejectedBeforeForward'
            before=None
            if not rejection:
                if previous is not None and row['timestamp_s']<=previous:raise RuntimeError('NonIncreasingModelTimestamp')
                before=memory(model)
                if before['history_timestamp'] is not None and any(t>=row['timestamp_s'] for t in before['history_timestamp']):raise RuntimeError('FutureModelMemory')
                with (root.parent/'forward_budget.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);entries=[json.loads(x) for x in ledger.read_text().splitlines()]
                    live=sum(x['kind']=='live_pipeline' for x in entries)
                    if len(entries)>=cfg['total_forward_cap'] or live>=cfg['max_forward_attempts']:rejection='ForwardBudgetEnded'
                    checked=age(row,start,wait_for_snapshot=True)
                    if not rejection and checked['reason']:rejection='InputRejectedBeforeForward'
                    if not rejection:
                        attempt=len(entries)+1;append(ledger,dict(attempt=attempt,kind='live_pipeline',request=name,input_frame=row['frame_id'],wall_s=time.monotonic()))
            if rejection:
                rejected+=1
                append(root/'worker_timing.jsonl',dict(request=name,status=rejection,age=checked,original_check=item.get('checked'),
                    read_ms=item['read_ms'],preprocessing=item.get('preprocessing'),start_wall_s=start,
                    wait_after_cpu_ms=max(0.,wait_done-item.get('prepared',wait_done))*1000))
                respond(name,rejection,reason=checked['reason'] or rejection,model_forward=False);continue
            selection.clear();torch.cuda.reset_peak_memory_stats();infer=time.monotonic()
            with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
            torch.cuda.synchronize();inferred=time.monotonic()
            if not finite(prediction):raise RuntimeError('NonfiniteNativeOutput')
            finite_done=time.monotonic();trajectory,meta=planning_to_hs(prediction,row['timestamp_s']);converted=time.monotonic()
            control=ctrl.command(prediction,row['speed_mps'],measured_navigation_target(row));finished=time.monotonic();result_age=age(row,start)
            reason,diagnostics=validate_trajectory(trajectory.points,row['speed_mps'],c['trajectory_checks'])
            if not finite(control) or not all(math.isfinite(float(x)) for part in ('raw_pid','native_agent_control') for x in control[part].values()):raise RuntimeError('NonfinitePIDOutput')
            if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('ParameterChanged')
            after=memory(model);checked_age=age(row,start);checked_done=time.monotonic()
            # No diagnostics/controls on IPC. Safety/finite/trajectory checks completed.
            respond(name,'ShadowComplete',model_forward=True,finite=True,result_expired=checked_age['reason'] is not None)
            published=time.monotonic()
            record=dict(status='ShadowDiagnostic',source='FrozenModelDryRun',attempt=attempt,frame=row['frame_id'],timestamp_s=row['timestamp_s'],
                input_wall_s=row['input_capture_wall_s'],forward_start_wall_s=infer,result_wall_s=finished,checked_wall_s=checked_done,published_wall_s=published,
                native_path=plain(prediction['lat_reg_final']),native_speeds=plain(prediction['lon_reg_final']),native_trajectory=plain(prediction['traj_final']),
                hs_points=trajectory.points.tolist(),hs_metadata=meta,raw_pid=plain(control['raw_pid']),native_agent_control=control['native_agent_control'],
                pid_metadata=plain(control['pid_metadata']),applied_to_vehicle=False,trajectory_check_reason=reason,trajectory_diagnostics=diagnostics,
                result_age_reason=result_age['reason'],completion_age=result_age,checked_age=checked_age,pre_forward_age=checked,
                memory_before=before,memory_after=after,timestamp_gap_s=None if previous is None else row['timestamp_s']-previous,
                preprocessing=item['preprocessing'],timings_ms=dict(ipc_read=item['read_ms'],cpu_preprocess=(item['prepared']-item['preprocess_start'])*1000,
                    wait_after_cpu=(wait_done-item['prepared'])*1000,gpu_transfer=transfer,prepare_thread_cpu=item['prepare_thread_cpu_ms'],
                    inference_decode_with_selection_audit=(inferred-infer)*1000,finite_check=(finite_done-inferred)*1000,
                    mapping=(converted-finite_done)*1000,pid=(finished-converted)*1000,necessary_checks=(checked_done-finished)*1000,
                    end_to_end_result=(finished-row['input_capture_wall_s'])*1000,end_to_end_checked=(checked_done-row['input_capture_wall_s'])*1000,
                    worker_to_result=(finished-start)*1000,completion_publish=(published-checked_done)*1000),
                gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved(),finite=True)
            # Synchronous bounded persistence: one immutable copied input and prediction.
            # This delays the next inference, is measured, and fails closed on write errors.
            saving=time.monotonic();torch.save(dict(prediction=prediction,selection=dict(selection)),root/'native'/(Path(name).stem+'.pt'))
            native_saved=time.monotonic();np.savez(root/'input_rows'/(Path(name).stem+'.npz'),**item['images']);saved=time.monotonic()
            record['timings_ms'].update(post_publish_diagnostics=(saving-published)*1000,native_save=(native_saved-saving)*1000,input_archive=(saved-native_saved)*1000)
            write(root/'diagnostics'/name,record);logged=time.monotonic()
            append(root/'worker_timing.jsonl',dict(request=name,status='ShadowComplete',attempt=attempt,start_wall_s=start,finish_wall_s=logged,
                diagnostic_json_write_ms=(logged-saved)*1000,post_publish_persistence_ms=(logged-published)*1000))
            completed+=1;previous=row['timestamp_s']
            del batch,prediction,item
    except BaseException as exc:atomic(root/'model_error.json',dict(error=repr(exc),attempt=attempt,traceback=traceback.format_exc()))
    finally:
        stop.set();pending.close()
        if thread is not None:thread.join(timeout=5.)
        unchanged=None
        if model is not None and baseline is not None:reset(model);unchanged=state_digest(model)==baseline
        atomic(root/'model_final.json',dict(completed_forwards=completed,input_rejections=rejected,fixed_state_unchanged=unchanged,
            preprocessing_thread_stopped=thread is None or not thread.is_alive(),model_controls=0,optimizer_steps=0,
            full_forward_attempts=len(ledger.read_text().splitlines()) if ledger.exists() else 0))

if __name__=='__main__':main()
