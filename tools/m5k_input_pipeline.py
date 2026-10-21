"""Measured complete offline and native-buffer replay pipelines, never driving."""
import json,time,io
from pathlib import Path
import torch,numpy as np,cv2
from tools.m5k_common import rows,write,asset,summary
from tools.m5f_common import inputs
from models.m5j_candidate import MobileBEVCandidate
from simulator.m5k_input import bgra_to_pinned,to_gpu,LatestRGBPacket
from training.runtime import prepare_cuda
from simulator.recorder import file_hash

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);row=rows('historical_val')[2];cpu=inputs(row,'cpu');metadata={k:v.pin_memory()for k,v in cpu.items()if k!='images'};pinned=(cpu['images']*255).round().byte().pin_memory();native=Path('runs/m5k_native_buffer');manifest=json.loads((native/'manifest.json').read_text());buffers=[]
 for r in manifest['buffers']:
  assert file_hash(r['path'])==r['sha256'];buffers.append(Path(r['path']).read_bytes())
 # Native packet calibration and time are recorded with the same six cameras.
 nmeta={k:torch.tensor(v['value'],dtype=getattr(torch,v['dtype'])).pin_memory()for k,v in manifest['metadata'].items()};records=[];raw=[]
 for seed in [101,102]:
  net=MobileBEVCandidate(enable=True,seed=seed)
  for mode in ['file_sha','memory_float','memory_pinned_uint8','native_bgra_pinned','torch_serialized']:
   times=[];stages=[];torch.cuda.reset_peak_memory_stats()
   for i in range(45):
    torch.cuda.synchronize();t=time.perf_counter();beg=t
    if mode=='file_sha':x=inputs(row,'cpu');t0=time.perf_counter();gpu={k:v.cuda()for k,v in x.items()}
    elif mode=='memory_float':x=cpu;t0=time.perf_counter();gpu={k:v.cuda()for k,v in x.items()}
    elif mode=='torch_serialized':
     b=io.BytesIO();torch.save(cpu,b);b.seek(0);x=torch.load(b,weights_only=True);t0=time.perf_counter();gpu={k:v.cuda()for k,v in x.items()}
    elif mode=='memory_pinned_uint8':t0=time.perf_counter();gpu=to_gpu(pinned,metadata)
    else:
     x=bgra_to_pinned(buffers);t0=time.perf_counter();gpu=to_gpu(x,nmeta)
    torch.cuda.synchronize();t1=time.perf_counter();out=net(now_s=float(gpu['timestamps_s'][0]),**gpu);torch.cuda.synchronize();end=time.perf_counter()
    if i>=5:times.append((end-beg)*1000);stages.append(dict(prepare_ms=(t0-beg)*1000,h2d_ms=(t1-t0)*1000,inference_ms=(end-t1)*1000,total_ms=(end-beg)*1000))
   raw.append(dict(seed=seed,mode=mode,samples=stages));records.append(dict(seed=seed,mode=mode,complete_pipeline_ms=summary(times),peak_mib=torch.cuda.max_memory_allocated()/2**20,over50ms=sum(t>50 for t in times)))
  del net;torch.cuda.empty_cache()
 packet=LatestRGBPacket();events=[]
 for label,frames,ts,now in [('valid',[10]*6,[1.]*6,1.1),('duplicate',[10]*6,[1.]*6,1.1),('stale',[11]*6,[2.]*6,2.3),('mismatch',[12]*5+[13],[3.]*6,3.1),('future',[14]*6,[4.]*6,3.9),('recovery',[15]*6,[5.]*6,5.1)]:
  try:packet.accept(frames,ts,now);status='accepted'
  except ValueError as e:status='rejected';reason=str(e)
  events.append(dict(label=label,status=status))
 write('runs/m5k_logs/input_timing_raw.json',raw);write('reports/m5k_input_pipeline.json',dict(kind='actual_RTX4090_measured_end_to_end_offline_and_actual_CARLA_buffer_replay',records=records,raw=asset('runs/m5k_logs/input_timing_raw.json'),native=asset(native/'manifest.json'),packet_faults=events,source_sha256=file_hash(__file__),limits='End-to-end walltime measured directly includesconversion/H2D/inputvalidation/full dualRGB+Lift+support. Warmup5then40; no sumP95. Native BGRA actualsensor bytes, replay not live queue/tick latency. Buffer copies owned before callbacks expire. Timestampfaults synthetic unit exercise. NoIPC transport, no perframe tensorserialization in proposed path. No20Hzdrivingclaim.'))
if __name__=='__main__':main()
