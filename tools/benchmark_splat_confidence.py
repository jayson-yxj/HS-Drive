"""M4.5 fixed-checkpoint A/B/C aggregation study; no training."""
import argparse,json,time,os,subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from diagnostics.splat_confidence import ConfidenceSplat,aggregate_sum,feature_buckets
from diagnostics.experiment import RUNS,road_confusion,read
from diagnostics.spatial import compare_mass
from training.runner import seed_all,initialize,load_checkpoint,to_device,road_metrics
from training.metrics import depth_prediction,DepthMetrics
from training.runtime import prepare_cuda
from datasets.training import HSDriveDataset,make_dataloader
from simulator.recorder import atomic_json,file_hash


def process_mib():
    try:
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
        return next(float(r.split(',')[1]) for r in rows if int(r.split(',')[0])==os.getpid())
    except (ValueError,StopIteration,subprocess.SubprocessError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m45_splat_comparison.json');p.add_argument('--artifacts',default='reports/m45_splat_figures');a=p.parse_args()
    root=Path(a.artifacts)
    if Path(a.output).exists() or root.exists():raise FileExistsError('New outputs required')
    root.mkdir();config=yaml.safe_load(Path('configs/m45_splat.yaml').read_text());seed_all(7);prepare_cuda();runs={}
    atomic_json(root/'protocol.json',dict(config=config,config_sha256=file_hash('configs/m45_splat.yaml'),scope='native all-ray frozen inference; common v1 road mask; geometry metrics copied from audited same-ray M4 experiment because mass and probability are invariant'))
    old=read('reports/m4_depth_distribution_diagnostics.json')
    modes=[('baseline','baseline',None),('sum','sum',None)]+[(f'gated_{t:g}','gated',t) for t in config['tau_candidates']]
    for name in RUNS:
        man=read(Path('runs')/name/'manifest.json');cfg=man['config'];model=initialize(cfg,pretrained=False).cuda().eval();load_checkpoint(Path('runs')/name/'best.pt',model)
        expected_hash=file_hash(Path('runs')/name/'best.pt');assert expected_hash==man['best_sha256']
        params={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};geometry=yaml.safe_load(Path(cfg['bev_config']).read_text())
        splats={key:ConfidenceSplat(geometry,mode=mode,tau=tau).cuda() for key,mode,tau in modes};ds=HSDriveDataset(cfg['index'],'val')
        rows=[];mass_by_scene={};metrics=DepthMetrics();conf={key:{} for key,_,_ in modes};perm=torch.randperm(960,generator=torch.Generator().manual_seed(91)).cuda();peak_process=None
        with torch.inference_mode():
            for i,batch in enumerate(make_dataloader(ds,workers=2)):
                inp,t=to_device(batch);sid=batch.metadata[0]['scene_id'];captured={}
                def hook(m,args,out):captured['args']=args
                handle=model.splat.register_forward_hook(hook);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):o=model(**inp.as_model_kwargs())
                torch.cuda.synchronize();full_ms=(time.perf_counter()-start)*1000;handle.remove();args=captured['args'];base=o['height_bev'];mass=o['splat_mass'];oracle=model.splat.observed_counts(t['depth_m'],t['depth_uv'],t['depth_valid'],inp['intrinsics'],inp['cam_to_ego'])
                metrics.update(depth_prediction(o,t,inp,model.splat.depth_centers),t['depth_m'],t['depth_valid'])
                mass_by_scene.setdefault(sid,[]).append(mass[0].cpu().numpy())
                rec={'sample_id':batch.metadata[0]['sample_id'],'scene':sid,'native_forward_ms':full_ms,'mass_total':float(mass.sum()),'mass_nonzero_voxels':int((mass>0).sum()),'mass_vs_oracle':compare_mass(mass,oracle),'variants':{}}
                for key,mode,tau in modes:
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    h,m=splats[key](*args);torch.cuda.synchronize();runtime=(time.perf_counter()-start)*1000
                    if not torch.allclose(m,mass,atol=2e-5,rtol=1e-6):raise ValueError('Aggregation changed mass')
                    if key=='baseline' and not torch.allclose(h,base,atol=2e-5,rtol=1e-6):raise ValueError('Default changed')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        bev=model.bev_encoder(torch.cat((h.flatten(1,2),torch.log1p(m)),1));road=model.drivable_head(bev)[:,0]
                        sh,sm=splats[key](args[0],args[1].flatten(-2)[...,perm].reshape_as(args[1]),*args[2:]);sb=model.bev_encoder(torch.cat((sh.flatten(1,2),torch.log1p(sm)),1));sr=model.drivable_head(sb)[:,0]
                    cm=road_confusion(road,t);conf[key].setdefault(sid,np.zeros((2,2),np.int64))[:]+=cm;rr=road_metrics(cm);tn,fp,fn,tp=cm.ravel();rr['precision_positive']=float(tp/(tp+fp)) if tp+fp else None
                    bucket=feature_buckets(h,m,oracle)
                    rec['variants'][key]={**bucket,'runtime_ms':runtime,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                        'BEV_relative_L1_change':float((bev.float()-o['bev_current'].float()).abs().sum()/o['bev_current'].float().abs().sum().clamp_min(1e-6)),
                        'BEV_feature_cosine_similarity':float(F.cosine_similarity(bev.float(),o['bev_current'].float(),dim=1).mean()),'drivable':rr,
                        'drivable_probability_change':float((road.float().sigmoid()-o['drivable_logits'].float().sigmoid()).abs()[t['drivable_valid']].mean()),
                        'depth_shuffle_probability_change':float((sr.float().sigmoid()-road.float().sigmoid()).abs()[t['drivable_valid']].mean()),
                        'depth_shuffle_road':road_metrics(road_confusion(sr,t))}
                    if i in (0,161):np.savez_compressed(root/(name+'_'+str(i)+'_'+key+'.npz'),mass=m[0].cpu().numpy(),feature_l1=h[0].abs().sum(0).cpu().numpy(),road=road[0].float().sigmoid().cpu().numpy(),oracle=oracle[0].cpu().numpy())
                if i%40==0:
                    peak=process_mib();peak_process=max(peak_process or 0,peak) if peak is not None else peak_process
                    print(name,i+1,len(ds),flush=True)
                rows.append(rec)
        quantiles={}
        for sid,values in mass_by_scene.items():
            ar=np.stack(values);quantiles[sid]={f'z{z}':{'all':np.quantile(ar[:,z],[0,.1,.25,.5,.75,.9,.99,1]).tolist(),'positive':np.quantile(ar[:,z][ar[:,z]>0],[0,.1,.25,.5,.75,.9,.99,1]).tolist()} for z in range(4)}
        original=old['results'][name];error=max(abs(metrics.result()[b]['mae_m']-original['distribution']['all_valid']['all'][b]['expected_mae_m']) for b in metrics.result())
        assert error<1e-4 and all(torch.equal(v.cpu(),params[k]) for k,v in model.state_dict().items())
        runs[name]={'frames':rows,'mass_quantiles':quantiles,'quantile_levels':[0,.1,.25,.5,.75,.9,.99,1],'process_gpu_memory_mib_sampled_peak':peak_process,
            'checkpoint_sha256':expected_hash,'reproduced_depth_mae_max_error':error,'weights_unchanged':True,
            'road':{k:{s:road_metrics(m) for s,m in sc.items()} for k,sc in conf.items()},
            'single_ray_geometry_invariant':original['distribution'],'matched_ray_bev_evidence':f'reports/m4_depth_distribution_artifacts/{name}.json',
            'note':'All A/B/C probability and mass metrics identical by construction; native all-ray mass comparison and matched-valid-ray metrics have different coverage and are explicitly separated.'}
        atomic_json(root/(name+'.json'),runs[name]);del model;torch.cuda.empty_cache()
    atomic_json(a.output,dict(schema_version='hsd-m45-splat-v1',status='complete',no_training=True,default_unchanged=True,config=config,results=runs,old_diagnostics_sha256=file_hash('reports/m4_depth_distribution_diagnostics.json')))

if __name__=='__main__':main()
