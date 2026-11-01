"""Prepared offline A2 checks only; not executed and no live inference service."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from tools.m8j12_runtime import build,reset,state_digest,observation_batch
from tools.m8j12_replay import finite,plain,install_selection_log
from datasets.m8j12_contract import official_camera_contract,planning_to_hs
from tools.m8j13_common import config,write,atomic,append,digest,provenance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--offline-a2-only',action='store_true');a=parser.parse_args();root=Path(a.root);c=config()
    if not a.offline_a2_only:raise RuntimeError('LiveWorkerDisabledUntilJ13AComplete')
    os.umask(0o077);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model,cfg,load=build();model.cuda();baseline=state_digest(model)
    versions={n:p._version for n,p in model.named_parameters()}
    camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    selection={};install_selection_log(model,selection)
    ledger=Path(c['private_root'])/'forward_budget.jsonl'
    count=len(ledger.read_text().splitlines()) if ledger.exists() else 0
    def run(row,images,out,kind):
        nonlocal count
        if count>=c['max_full_forward_attempts']:raise RuntimeError('ForwardBudgetExhausted')
        start=time.monotonic();batch=observation_batch(cfg,row,images,camera);torch.cuda.synchronize();prepared=time.monotonic()
        count+=1;append(ledger,dict(attempt=count,kind=kind,sequence=row['sequence_id'],input_frame=row['frame_id'],input_sim_s=row['timestamp_s'],wall_s=start))
        selection.clear();torch.cuda.reset_peak_memory_stats()
        inference_start=time.monotonic()
        with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
        torch.cuda.synchronize();inference_end=time.monotonic()
        if not finite(prediction):raise RuntimeError('NonfiniteNativeOutput')
        tr,meta=planning_to_hs(prediction,row['timestamp_s']);converted=time.monotonic()
        if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('ParameterVersionChanged')
        public=dict(sequence=row['sequence_id'],input_frame=row['frame_id'],input_sim_s=row['timestamp_s'],input_wall_s=row['input_capture_wall_s'],
            inference_start_wall_s=start,model_start_wall_s=inference_start,output_wall_s=converted,
            model_sha256='f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130',
            path=plain(prediction['lat_reg_final']),speeds=plain(prediction['lon_reg_final']),native_trajectory=plain(prediction['traj_final']),
            hs_points=tr.points.tolist(),hs_meta=meta,selected_index=selection.get('traj_mode_idx'),
            output_keys=sorted(prediction),input_pose=row['ego_world_matrix'],input_speed=row['speed_mps'],
            timings_ms=dict(preprocess=(prepared-start)*1000,model_and_decode=(inference_end-inference_start)*1000,trajectory_conversion=(converted-inference_end)*1000,total_compute=(converted-start)*1000),
            gpu_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.save(dict(prediction=prediction,selection=selection,hs_points=tr.points,metadata=public),out/'native.pt')
        public['published_wall_s']=time.monotonic();public['status']='Passed';atomic(out/'response.json',public)
        return public
    try:
        # Four fixed old inputs: two scene starts, each independently reset twice.
        capture=json.loads(Path('reports/m8j12_sources/capture_result.json').read_text());old=Path(capture['observation_root']);pairs={};checks=[]
        for replay in range(2):
            for sequence in ('S0','S1'):
                reset(model);row=json.loads((old/sequence/'manifest.json').read_text())[0]
                row['sequence_id']='A2_'+sequence;row['input_capture_wall_s']=time.monotonic()
                images=dict(np.load(old/sequence/row['images_file'],allow_pickle=False));out=root/('a2_'+sequence+'_'+str(replay));out.mkdir()
                result=run(row,images,out,'A2_prior_recorded_input')
                if replay:
                    for key in ('path','speeds','native_trajectory','hs_points'):
                        x=np.asarray(result[key]);y=np.asarray(pairs[sequence][key]);assert np.allclose(x,y,atol=1e-5,rtol=1e-5)
                    checks.append(dict(sequence=sequence,max_abs=max(float(np.max(np.abs(np.asarray(result[k])-np.asarray(pairs[sequence][k])))) for k in ('path','speeds','native_trajectory','hs_points'))))
                else:pairs[sequence]=result
        reset(model);assert state_digest(model)==baseline
        atomic(root/'ready.json',dict(status='A2Passed',load=load,checks=checks,full_forward_attempts=count,provenance=provenance(),fixed_state_unchanged=True))
        if a.offline_a2_only:return
    finally:
        unchanged=state_digest(model)==baseline
        write(root/'final.json',dict(full_forward_attempts=count,fixed_state_unchanged=unchanged,optimizer_steps=0,provenance=provenance()))


if __name__=='__main__':main()
