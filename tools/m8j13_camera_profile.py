"""Fixed first8 actual inputs,3 old/new preprocessing replays each. No model.

Release each historical camera at its original copy-complete offset. This is
an offline scheduling experiment, not a new sensor capture or online freshness.
"""
import json,os,time
from pathlib import Path
import numpy as np
import torch,cv2
from mmcv import Config
import projects.mmdet3d_plugin
from datasets.m8j12_contract import CAMERAS,official_camera_contract
from tools.m8j12_runtime import protocol
from tools.m8j13_preprocess import OfficialPreprocessor,equal_nested
from tools.m8j13_camera_preprocess import CameraPreprocessor
from simulator.m8j13_transport import Slots
from simulator.m8j13_partial_transport import PartialSlots
from tools.m8j13_common import write,append
from tools.m8j13_camera_timeline import ROOT,OLD


def wait_until(t):
    delay=t-time.monotonic()
    if delay>0:time.sleep(delay)


def main():
    timelines=json.loads((ROOT/'historical_timelines.json').read_text());actual=[x for x in timelines if 'diagnostic' in x][:8]
    write(ROOT/'profile_request.json',dict(inputs=[x['candidate']['request'] for x in actual],repeats=3,max_preprocess_calls=48,full_forwards=0,
        affinity=sorted(os.sched_getaffinity(0)),cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),torch_threads=torch.get_num_threads(),
        opencv_threads=cv2.getNumThreads(),torch_interop=torch.get_num_interop_threads(),candidate_workers=2,
        schedule='Real camera copy-complete offsets; group commit at original owner validation time; original row times retained. Replay clock only schedules sleep; never substituted into model metadata.',
        loading='NPZ input read before timing; BGRA reconstructed with alpha0, ignored by both unchanged BGR extraction paths. Original recorded raw BGRA not archived; no claim of exact alpha bytes.'))
    c=protocol();rt=Path(c['runtime_root']);cfg=Config.fromfile(str(rt/'runtime-source'/c['config']));camera=official_camera_contract(rt/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    oldprep=OfficialPreprocessor(cfg,camera);newprep=CameraPreprocessor(cfg,camera)
    whole=Slots(Path('/dev/shm')/('hs-j13-camera-old-'+str(os.getpid())),create=True)
    partial=PartialSlots(Path('/dev/shm')/('hs-j13-camera-new-'+str(os.getpid())),create=True);records=[]
    try:
        for data in actual:
            name=data['candidate']['request'];row=json.loads((OLD/'input_rows'/name).read_text());images=dict(np.load((OLD/'input_rows'/name).with_suffix('.npz'),allow_pickle=False))
            raw={}
            for n in CAMERAS:
                bgra=np.zeros((396,704,4),np.uint8);bgra[:,:,:3]=images[n];raw[n]=bgra.tobytes()
            stamps=sorted(data['cameras'],key=lambda x:x['copied_monotonic_s']);identity=dict(run=stamps[0]['run'],episode=stamps[0]['episode'],generation=stamps[0]['sensor_generation'],sensors={x['camera']:x['actor_id'] for x in stamps})
            origin=data['origin'];group_at=data['candidate']['decision_wall_s']+data['candidate']['group_checks_ms']/1000
            for rep in range(3):
                pair={};batches={}
                for mode in ('old','candidate'):
                    append(ROOT/'preprocess_budget.jsonl',dict(input=name,repeat=rep,path=mode))
                    request='%s-%s-%d'%(name,mode,rep);start=time.monotonic();cpu=time.process_time()
                    if mode=='old':
                        wait_until(start+group_at-origin);begin=time.monotonic()
                        token=whole.publish(request,row,raw,identity);published=time.monotonic();r,im,_=whole.read(token);read=time.monotonic()
                        batch,timing=oldprep.cpu(r,im);prepared=time.monotonic();batch,gpu=oldprep.gpu(batch);finished=time.monotonic()
                        pair[mode]=dict(end_to_batch_ms=(finished-start)*1000,process_cpu_ms=(time.process_time()-cpu)*1000,
                            group_wait_ms=(begin-start)*1000,publish_ms=(published-begin)*1000,read_ms=(read-published)*1000,
                            preprocessing=timing,cpu_preprocess_ms=(prepared-read)*1000,gpu_ms=gpu)
                        whole.release(request)
                    else:
                        token=partial.begin(request,identity,row['frame_id'],row['timestamp_s'],row['input_capture_wall_s']);newprep.begin(token);publish=[]
                        for stamp in stamps:
                            wait_until(start+stamp['copied_monotonic_s']-origin);p=time.monotonic()
                            partial.publish_camera(token,stamp,raw[stamp['camera']]);newprep.submit(partial,token,stamp['camera']);publish.append((time.monotonic()-p)*1000)
                        wait_until(start+group_at-origin);committing=time.monotonic()
                        partial.commit(token,row,data['snapshot'],group_at,row['timestamp_s']+data['candidate']['sim_age_s']);committed=time.monotonic()
                        batch,timing=newprep.assemble(partial,token);prepared=time.monotonic();batch,gpu=newprep.gpu(batch);finished=time.monotonic()
                        pair[mode]=dict(end_to_batch_ms=(finished-start)*1000,process_cpu_ms=(time.process_time()-cpu)*1000,
                            per_camera_publish_ms=publish,commit_ms=(committed-committing)*1000,group_ready_ms=(committed-start)*1000,
                            assembled_ms=(prepared-start)*1000,gpu_ms=gpu,detail=timing)
                        newprep.finish(token);partial.release(request)
                    batches[mode]=batch
                equal_nested(batches['old'],batches['candidate'])
                suffix_ms=(data['diagnostic']['checked_wall_s']-data['diagnostic']['forward_start_wall_s'])*1000
                # Original pre-forward check/ledger time retained rather than set to zero.
                oldd=data['diagnostic'];oldw=data['worker'];pre_ms=oldd['timings_ms']['worker_to_result']-oldd['timings_ms']['ipc_read']-oldd['timings_ms']['cpu_preprocess']-oldd['timings_ms']['wait_after_cpu']-oldd['timings_ms']['gpu_transfer']-(oldd['result_wall_s']-oldd['forward_start_wall_s'])*1000
                pair.update(input=name,repeat=rep,exact_equal=True,critical_path_saved_ms=pair['old']['end_to_batch_ms']-pair['candidate']['end_to_batch_ms'],
                    counterfactual_checked_ms=pair['candidate']['end_to_batch_ms']+suffix_ms+max(0,pre_ms),historical_remaining_compute_and_checks_ms=suffix_ms+max(0,pre_ms))
                records.append(pair);append(ROOT/'profile_records.jsonl',pair);del batches
        write(ROOT/'profile_result.json',dict(status='ExactInputEquivalencePassed',records=records,full_forwards=0))
        print('24/24 exact input pairs passed')
        for mode in ('old','candidate'):
            print(mode,'end_to_batch_ms',np.percentile([x[mode]['end_to_batch_ms'] for x in records],[0,50,95,100]),'cpu_ms',np.percentile([x[mode]['process_cpu_ms'] for x in records],[50,95]))
        print('counterfactual subsequent samples',np.percentile([x['counterfactual_checked_ms'] for x in records if x['input']!=records[0]['input']],[0,50,95,100]))
    finally:newprep.close();partial.close();whole.close()

if __name__=='__main__':main()
