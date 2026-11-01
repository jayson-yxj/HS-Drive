"""Actual3.10 relay ->3.8 preparation -> completion consumer, no CARLA/model.

Historical timestamps remain unchanged. Explicit synthetic clock only tests the
age contract; this is never online freshness or measured input performance.
"""
import argparse,json,os,subprocess,threading,time
from pathlib import Path
from queue import Empty
import numpy as np
from tools.m8j13_common import atomic,write
from simulator.m8j13_snapshot_age import SnapshotAge,consumed_completion


def child(root):
    from mmcv import Config
    import projects.mmdet3d_plugin
    from tools.m8j12_runtime import protocol
    from datasets.m8j12_contract import official_camera_contract
    from tools.m8j13_camera_live import camera_prepare_loop
    from simulator.m8j13_prepared_slot import PreparedSlot
    c=protocol();rt=Path(c['runtime_root']);cfg=Config.fromfile(str(rt/'runtime-source'/c['config']))
    camera=official_camera_contract(rt/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    age=SnapshotAge(root,clock=lambda:json.loads((root/'test_clock.json').read_text())['now'])
    def historical_age(row,start,**kw):return age(row,row['input_capture_wall_s'],**kw)
    pending=PreparedSlot();stop=threading.Event();thread=threading.Thread(target=camera_prepare_loop,args=(root,cfg,camera,pending,stop,historical_age));thread.start()
    atomic(root/'ready.json',dict(ready=True,full_forwards=0))
    try:
        until=time.monotonic()+20;done=0
        while done<2:
            if time.monotonic()>until:raise TimeoutError('BoundaryWorkerTimeout')
            try:item=pending.take(.05)
            except Empty:continue
            if 'error' in item:raise RuntimeError(item)
            if item.get('cancelled'):response=dict(status='PartialCancelled',reason='OwnerInputRejected',model_forward=False)
            else:
                if 'rejection' in item:raise RuntimeError(item['rejection'])
                assert set(item['images'])==set(camera['sensors'])
                assert item['cpu_batch']['img'].data[0].shape==(1,6,3,384,704)
                # Real3.8 age reader observes pending -> confirmed while waiting.
                row=item['row'];frame=row['frame_id'];wall=row['input_capture_wall_s'];sim=row['timestamp_s']
                atomic(root/'tick_pending.json',dict(frame=frame+1,pending=True));atomic(root/'test_clock.json',dict(now=wall+.2))
                pending_result=historical_age(row,wall,phase='integration_pending')
                assert pending_result['reason']=='SimulationSnapshotUnconfirmed' and pending_result['simulation_expired'] is None
                def confirm():
                    time.sleep(.02);atomic(root/'current_snapshot.json',dict(frame=frame+1,sim_s=sim+.05))
                    atomic(root/'test_clock.json',dict(now=wall+.22));atomic(root/'tick_pending.json',dict(frame=frame+1,pending=False))
                confirmation=threading.Thread(target=confirm);confirmation.start()
                resumed=historical_age(row,wall,wait_for_snapshot=True,phase='integration_pending_to_confirmed');confirmation.join()
                assert resumed['reason'] is None and resumed['snapshot_wait_ms']>0
                atomic(root/'test_clock.json',dict(now=wall+.31));expired=historical_age(row,wall,wait_for_snapshot=True,phase='integration_already_expired')
                assert expired['reason']=='PlanWallExpired' and expired['snapshot_wait_ms']==0
                response=dict(status='BoundaryPrepared',model_forward=False,finite=True,trajectory_valid=True)
                write(root/'prepared_summary.json',dict(dtype=str(item['cpu_batch']['img'].data[0].dtype),camera_count=len(item['images']),same_frame_imu=item['row']['imu_frame']==item['row']['frame_id'],navigation_target=item['row']['target_world_xy'],no_model=True))
            atomic(root/'responses'/item['name'],response);done+=1
        (root/'model_stop').touch();stop.set();pending.close();thread.join(5)
        assert not thread.is_alive()
        atomic(root/'child_final.json',dict(preparation_thread_stopped=True,full_forwards=0,carla_imported='carla' in __import__('sys').modules))
    finally:stop.set();pending.close();thread.join(5)


def parent(root):
    from tools.m8j13_camera_live import EarlyRelay
    from simulator.m8j13_candidate_audit import CandidateInbox,CopiedImage
    from simulator.m8j13_sensor_audit import RawImageStamp
    from simulator.m8j13_partial_transport import PartialSlots
    from datasets.m8j12_contract import CAMERAS
    old=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101/live');path=sorted((old/'diagnostics').glob('*.json'))[0]
    row=json.loads((old/'input_rows'/path.name).read_text());images=dict(np.load((old/'input_rows'/path.name).with_suffix('.npz'),allow_pickle=False))
    snaps=[json.loads(x) for x in (old/'snapshots.jsonl').read_text().splitlines()];snapshot=next(x for x in snaps if x['frame']==row['frame_id'])
    cams=[json.loads(x) for x in (old/'camera_metadata.jsonl').read_text().splitlines() if json.loads(x)['frame']==row['frame_id']]
    identity=dict(run=cams[0]['run'],episode=cams[0]['episode'],generation=cams[0]['sensor_generation'],sensors={x['camera']:x['actor_id'] for x in cams})
    root.mkdir();(root/'input_rows').mkdir();(root/'responses').mkdir();slots=PartialSlots(Path('/dev/shm')/('hs-j13-recheck-boundary-'+str(os.getpid())),create=True)
    source=CandidateInbox((identity['run'],identity['episode'],identity['generation']));relay=EarlyRelay(source,slots,identity,root);process=None
    def payload(meta):
        n=meta['camera'];matrix=tuple(tuple(x for x in y) for y in meta['transform']);stamp=RawImageStamp(identity['run'],identity['episode'],identity['generation'],n,meta['actor_id'],meta['frame'],meta['timestamp_s'],matrix,meta['receive_monotonic_s'])
        raw=np.zeros((396,704,4),np.uint8);raw[:,:,:3]=images[n]
        return CopiedImage(stamp,704,396,raw.tobytes(),meta['copied_monotonic_s'])
    def wait(predicate):
        until=time.monotonic()+20
        while not predicate():
            if process is not None and process.poll() is not None:raise RuntimeError('BoundaryChildExited')
            if time.monotonic()>until:raise TimeoutError('BoundaryWait')
            time.sleep(.002)
    try:
        atomic(root/'transport.json',dict(root=str(slots.root)))
        atomic(root/'current_snapshot.json',dict(frame=row['frame_id'],sim_s=row['timestamp_s']))
        atomic(root/'tick_pending.json',dict(pending=False,frame=row['frame_id']))
        atomic(root/'test_clock.json',dict(now=row['input_capture_wall_s']+.2))
        with (root/'child.log').open('x') as log:
            process=subprocess.Popen(['taskset','-c','11-12','bash','tools/m8j12_env.sh','-m','tools.m8j13_recheck_boundary','--child','--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
        wait(lambda:(root/'ready.json').exists())
        relay.reserve_frame(row['frame_id'],row['input_capture_wall_s'],0)
        for meta in cams:source.queue.put(payload(meta))
        wait(lambda:len(relay.queue.queue)==6)
        for copied in relay.drain():row['camera_world_matrices'][copied.stamp.camera]=copied.stamp.transform
        assert isinstance(row['camera_world_matrices'][CAMERAS[0]],tuple)
        # Whole actual relay commit checks same-frame snapshot, IMU, navigation row.
        relay.commit(row['frame_id'],row,snapshot,row['input_capture_wall_s']+.2,row['timestamp_s'])
        wait(lambda:(root/'responses'/'000000.json').exists());resp=json.loads((root/'responses'/'000000.json').read_text())
        consumed=consumed_completion(resp,row,row['frame_id'],row['timestamp_s'],row['input_capture_wall_s']+.21,row['input_capture_wall_s']+.22)
        assert not consumed['consumable_fresh'] # preparation is not a full model result
        relay.release('000000.json')
        # Reuse slot for real partial payload then cancellation, no new complete preprocess.
        frame=row['frame_id']+4;relay.reserve_frame(frame,row['input_capture_wall_s']+.2,1)
        meta=dict(cams[0],frame=frame,timestamp_s=row['timestamp_s']+.2);source.queue.put(payload(meta));wait(lambda:relay.frames[frame]['status']=='Partial')
        relay.reject(frame);wait(lambda:(root/'responses'/'000001.json').exists());assert json.loads((root/'responses'/'000001.json').read_text())['status']=='PartialCancelled';relay.release('000001.json')
        process.wait(timeout=10);assert process.returncode==0
        final=json.loads((root/'child_final.json').read_text());assert final['preparation_thread_stopped'] and not final['carla_imported']
        result=dict(status='Passed',creator_python='3.10',reader_python='3.8',actual_EarlyRelay=True,actual_prepare_loop=True,actual_CPU_pipeline=True,actual_consumer=True,
            tuple_JSON_binding=True,complete_inputs=1,partial_cancelled=1,slot_reused=True,full_model_forwards=0,carla_starts=0,
            clock_scope='Historical input unchanged; explicit synthetic clock for engineering age-contract test, no live freshness claim',child_clean_exit=True,pending_confirmed_expired_in_actual_reader=True)
    finally:
        (root/'model_stop').touch(exist_ok=True)
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=10)
        relay.close();slots.close()
    result['relay_thread_stopped']=not relay.thread.is_alive();result['slots_removed']=not slots.root.exists();write(root/'result.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:child(a.root)
    else:parent(a.root)
