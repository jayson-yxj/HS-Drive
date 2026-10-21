"""Independent shadow worker: deliberately no CARLA imports, tick or control calls."""
import time,queue

def latest_put(channel,value):
    try:channel.put_nowait(value);return True
    except queue.Full:
        try:channel.get_nowait()
        except queue.Empty:return False
        try:channel.put_nowait(value);return True
        except queue.Full:return False

def worker(channel,results,stop,ready):
    import torch
    from models.m5j_candidate import MobileBEVCandidate
    from simulator.m5k_input import bgra_to_pinned,to_gpu,LatestRGBPacket
    from training.runtime import prepare_cuda
    prepare_cuda();torch.set_num_threads(2);model=MobileBEVCandidate(enable=True,seed=101);packet=LatestRGBPacket();ready.set()
    while not stop.is_set():
        try:item=channel.get(timeout=.1)
        except queue.Empty:continue
        while True:
            try:item=channel.get_nowait()
            except queue.Empty:break
        start=time.perf_counter();age=start-item['captured_wall'];record=dict(frame=item['frame'],timestamp_s=item['timestamp_s'],queue_age_ms=age*1000)
        if age>.25:latest_put(results,record|dict(status='stale_before_compute'));continue
        try:
            packet.accept([item['frame']]*6,[item['timestamp_s']]*6,item['timestamp_s']+age)
            t=time.perf_counter();p=bgra_to_pinned(item['buffers']);meta={k:torch.tensor(v['value'],dtype=getattr(torch,v['dtype'])).pin_memory()for k,v in item['metadata'].items()};meta['timestamps_s']=torch.tensor([item['timestamp_s']],dtype=torch.float64).pin_memory();gpu=to_gpu(p,meta);torch.cuda.synchronize();h2d=(time.perf_counter()-t)*1000;t=time.perf_counter()
            with torch.no_grad():output=model(now_s=item['timestamp_s']+age,**gpu)
            torch.cuda.synchronize();infer=(time.perf_counter()-t)*1000;t=time.perf_counter();high=int((output['evidence']['scores']['raw']>=.2).sum());out_ms=(time.perf_counter()-t)*1000;end=time.perf_counter();total=end-item['captured_wall'];record.update(status='accepted'if total<=.25 else'expired_output_discarded',convert_h2d_ms=h2d,inference_ms=infer,output_ms=out_ms,end_to_end_ms=total*1000,high_cells=high)
        except Exception as e:record.update(status='rejected',error=repr(e))
        latest_put(results,record)
