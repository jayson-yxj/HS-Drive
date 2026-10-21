"""Complete Candidate compute incl support accounting; CPU/I-O separately measured."""
import io,json,time
from pathlib import Path
import torch,numpy as np
from models.m5j_candidate import MobileBEVCandidate
from tools.m5j_common import rows,write
from tools.m5f_common import inputs
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    from tools.m5j_freeze import verify
    verify();prepare_cuda();torch.set_num_threads(2);row=rows('test')[2];costs={'cpu_read_decode_validate':[],'serialize':[],'deserialize':[],'h2d':[]};records=[]
    for i in range(20):
        t=time.perf_counter();x=inputs(row,'cpu');costs['cpu_read_decode_validate'].append((time.perf_counter()-t)*1e3);t=time.perf_counter();buf=io.BytesIO();torch.save(x,buf);costs['serialize'].append((time.perf_counter()-t)*1e3);size=buf.tell();t=time.perf_counter();buf.seek(0);y=torch.load(buf,weights_only=True);costs['deserialize'].append((time.perf_counter()-t)*1e3);torch.cuda.synchronize();t=time.perf_counter();gpu={k:v.cuda()for k,v in y.items()};torch.cuda.synchronize();costs['h2d'].append((time.perf_counter()-t)*1e3)
    for visual in ['mobile_gt','legacy']:
        for seed in [101,102]:
            net=MobileBEVCandidate(enable=True,visual=visual,seed=seed);times=[];torch.cuda.reset_peak_memory_stats()
            for i in range(35):
                torch.cuda.synchronize();t=time.perf_counter();out=net(**gpu);torch.cuda.synchronize()
                if i>=5:times.append((time.perf_counter()-t)*1e3)
            records.append(dict(visual=visual,seed=seed,times_ms=times,p50=float(np.median(times)),p95=float(np.quantile(times,.95)),max=float(max(times)),peak_mib=torch.cuda.max_memory_allocated()/2**20));del net,out;torch.cuda.empty_cache()
    write('reports/m5j_latency.json',dict(kind='actual_RTX4090_complete_offline_dualbranch',records=records,cpu_costs={k:dict(times_ms=v,p50=float(np.median(v)),p95=float(np.quantile(v,.95)))for k,v in costs.items()},serialized_bytes=size,source_sha256=file_hash(__file__),limits='Includesinputvalidation+Pixel+Depth+exactsupport+Lift; excludesCARLA/controller. CPUtiming warmfilesystem withSHA validation. Serialization CPU tensor torchsave, not asserted networkIPC. No20Hzclosedloopclaim.'))
if __name__=='__main__':main()
