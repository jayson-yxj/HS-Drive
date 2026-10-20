"""Matched-ray Oracle feature comparison and isolated GPU process memory probe."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np,torch
from diagnostics.experiment import RUNS,flat_probs,sample_context
from diagnostics.spatial import project,compare_features,compare_mass
from diagnostics.splat_confidence import aggregate_sum
from models.view_transform import unproject
from datasets.training import HSDriveDataset,make_dataloader
from training.runner import initialize,load_checkpoint,seed_all,to_device
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):float(row.split(',')[1]) for row in output.splitlines() if row.strip()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m45_splat_oracle_probe.json');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    before=processes();seed_all(7);prepare_cuda();rows=[];memory=[]
    modes=[('baseline',None),('sum',None)]+[('gated',t) for t in (1e-4,1e-3,1e-2,.1)]
    for name in RUNS:
        man=json.loads((Path('runs')/name/'manifest.json').read_text());model=initialize(man['config'],pretrained=False).cuda().eval();load_checkpoint(Path('runs')/name/'best.pt',model)
        ds=HSDriveDataset(man['config']['index'],'val');torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch in make_dataloader(torch.utils.data.Subset(ds,[0,161]),workers=0):
                inp,t=to_device(batch);captured={}
                handle=model.context_head.register_forward_hook(lambda m,args,out:captured.update(context=out))
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inp.as_model_kwargs())
                finally:handle.remove()
                valid=t['depth_valid'];uv=t['depth_uv'];gt=unproject(uv,t['depth_m'],inp['intrinsics'],inp['cam_to_ego'])[valid]
                ctx=sample_context(captured['context'].reshape(1,6,64,24,40),uv,inp['images'].shape[-2:])[valid]
                probs=flat_probs(out,t,inp);d=model.splat.depth_centers.view(1,1,48,1,1).expand(1,6,48,24,40)
                bp=unproject(uv[:,:,None].expand(1,6,48,24,40,2),d,inp['intrinsics'],inp['cam_to_ego']).permute(0,1,3,4,2,5)[valid]
                h,m=project(ctx,probs,bp,model.splat);oh,om=project(ctx,torch.ones(len(gt),1,device='cuda'),gt[:,None],model.splat)
                weighted=h*m.clamp_min(1e-6)[:,None];oracle_weighted=oh*om.clamp_min(1e-6)[:,None]
                for mode,tau in modes:
                    hh=aggregate_sum(weighted,m,mode,tau);oo=aggregate_sum(oracle_weighted,om,mode,tau)
                    rows.append(dict(run=name,sample_id=batch.metadata[0]['sample_id'],mode=mode,tau=tau,
                        vs_fixed_oracle_mean=compare_features(hh,oh,m,om),vs_same_aggregation_oracle=compare_features(hh,oo,m,om),
                        layer_vs_fixed_oracle=[compare_features(hh[:,:,z:z+1],oh[:,:,z:z+1],m[:,z:z+1],om[:,z:z+1]) for z in range(4)],
                        mass=compare_mass(m,om)))
                torch.cuda.synchronize();current=processes();new=set(current)-set(before)
                memory.append(dict(run=name,sample_id=batch.metadata[0]['sample_id'],namespace_pid=__import__('os').getpid(),nvml_host_pids=list(current),new_host_pids=list(new),
                    own_process_mib=current[next(iter(new))] if len(new)==1 else None,
                    cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20))
        del model;torch.cuda.empty_cache()
    atomic_json(a.output,dict(schema_version='hsd-m45-splat-oracle-probe-v1',sampling='Predeclared first frame of each validation scene: indices 0,161; not full validation or performance selection',results=rows,memory_samples=memory,
        checkpoint_sha256={n:file_hash(Path('runs')/n/'best.pt') for n in RUNS},
        limitations=['Original full benchmark process memory is null: NVML host PID differs from container PID. This isolated probe identifies the unique new host PID; not a retrospective peak of full run.',
        'Oracle and learned use identical sampled valid rays and context. Fixed-oracle L1 can favor attenuation even without geometry improvement; compare cosine, mass and per-ray probability.',
        'GPU memory samples include all live diagnostic tensors and caching allocator; not a deployment-only benchmark.']))


if __name__=='__main__':main()
