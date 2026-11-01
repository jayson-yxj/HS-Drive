"""Historical first4 old/new paths,8 total forwards. Never online freshness."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from tools.m8j12_runtime import build,reset,state_digest,observation_batch,protocol
from tools.m8j12_replay import finite,plain,memory
from tools.m8j13_preprocess import OfficialPreprocessor
from simulator.m8j13_transport import Slots
from datasets.m8j12_contract import CAMERAS,official_camera_contract,planning_to_hs,DryRunController,measured_navigation_target
from tools.m8j13_common import write,append
ROOT=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101')
OLD=Path('/root/gpufree-data/hs-drive-private/m8j13-parked-shadow-20261101/live')

def main():
    write(ROOT/'equivalence_request.json',dict(first_inputs=[x.name for x in sorted((OLD/'diagnostics').glob('*.json'))[:4]],limit=8,scope='HistoricalInputEquivalence_NotOnlineFreshness'))
    c=protocol();rt=Path(c['runtime_root']);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model,cfg,_=build();model.cuda();reset(model);baseline=state_digest(model)
    camera=official_camera_contract(rt/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py');prep=OfficialPreprocessor(cfg,camera)
    outputs={};fixed=sorted((OLD/'diagnostics').glob('*.json'))[:4]
    slots=Slots(Path('/dev/shm/hs_m8j13_equivalence_'+str(os.getpid())),create=True)
    try:
        for mode in ['old','candidate']:
            reset(model);ctrl=DryRunController(rt/'official-source/leaderboard/team_code/pid_controller.py');out=[]
            for i,f in enumerate(fixed):
                row=json.loads((OLD/'requests'/f.name).read_text());images=dict(np.load((OLD/'requests'/f.name).with_suffix('.npz'),allow_pickle=False))
                if mode=='old':batch=observation_batch(cfg,row,images,camera)
                else:
                    raw={}
                    for n in CAMERAS:
                        bgra=np.zeros((396,704,4),np.uint8);bgra[:,:,:3]=images[n];raw[n]=bgra.tobytes()
                    token=slots.publish(f.name,row,raw,dict(run=row['sequence_id'],episode='historical',generation='equivalence'))
                    copied,images2,_=slots.read(token)
                    assert copied==row
                    for n in CAMERAS:assert np.array_equal(images[n],images2[n])
                    cpu,t=prep.cpu(copied,images2);batch,_=prep.gpu(cpu);slots.release(f.name)
                before=memory(model);count=len((ROOT/'forward_budget.jsonl').read_text().splitlines()) if (ROOT/'forward_budget.jsonl').exists() else 0
                if count>=8:raise RuntimeError('OfflineForwardBudgetExceeded')
                append(ROOT/'forward_budget.jsonl',dict(attempt=count+1,kind='offline_'+mode,input=f.name,wall_s=time.monotonic()))
                with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
                torch.cuda.synchronize();assert finite(prediction)
                trajectory,meta=planning_to_hs(prediction,row['timestamp_s']);control=ctrl.command(prediction,row['speed_mps'],measured_navigation_target(row))
                out.append(dict(native={k:plain(prediction[k]) for k in ['lat_reg_final','lon_reg_final','traj_final']},hs=trajectory.points.tolist(),meta=meta,pid=plain(control),memory_before=before,memory_after=memory(model),timestamp=row['timestamp_s']))
            outputs[mode]=out
        maxima={}
        for i,(a,b) in enumerate(zip(outputs['old'],outputs['candidate'])):
            assert a['timestamp']==b['timestamp'] and a['memory_before']==b['memory_before'] and a['memory_after']==b['memory_after']
            for k in ['lat_reg_final','lon_reg_final','traj_final']:
                x=np.asarray(a['native'][k]);y=np.asarray(b['native'][k]);np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-5);maxima[k]=max(maxima.get(k,0),float(np.max(abs(x-y))))
            np.testing.assert_allclose(a['hs'],b['hs'],rtol=1e-5,atol=1e-5)
            assert a['meta']==b['meta'] and a['pid']==b['pid']
        reset(model);assert state_digest(model)==baseline
        write(ROOT/'equivalence_outputs.json',outputs)
        write(ROOT/'equivalence_result.json',dict(status='Passed',forwards=8,failed=0,native_max_abs_difference=maxima,hs_equal=True,pid_equal=True,weights_unchanged=True,history_timestamps_preserved=True,scope='HistoricalOnly_NotOnlineFreshness'))
        print('8/8 forwards; native/HS/PID equivalence passed',maxima)
    finally:slots.close()

if __name__=='__main__':main()
