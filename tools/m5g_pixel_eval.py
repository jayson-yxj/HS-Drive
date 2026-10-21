"""Frozen candidate evaluation: single RGB frame; paired references offline only."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from tools.m5g_common import load_pixel
from tools.m5f_common import inputs,pixel_label
from tools.m5f_dataset import ROOT
from models.m5f_pixel import ForegroundModel
from tools.m5g_evaluation import summarize
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--split',default='val');a=p.parse_args();dest=Path(f'reports/m5g_pixel_{a.split}.json')
    if dest.exists():raise FileExistsError(dest)
    if a.split=='test':
        from tools.m5g_common import verify_freeze
        verify_freeze()
    prepare_cuda();torch.set_num_threads(2);rows=[r for r in json.loads(Path(a.root,'index.json').read_text())['records']if r['split']==a.split];labels=[pixel_label(r)for r in rows];study=json.loads(Path('reports/m5g_backbone_ablation.json').read_text())['records'];old=[dict(r,name='M5F_'+r['name'])for r in json.loads(Path('reports/m5f_image_foreground_training.json').read_text())['records']if r['name']=='P8_fpn'];study=old+study+json.loads(Path('reports/m5g_budget_extension.json').read_text())['records'];out=Path(f'runs/m5g_pixel_{a.split}');out.mkdir(exist_ok=False);results=[]
    for record in study:
        net,head,ck=load_pixel(record);model=ForegroundModel(net.encoder,head,8).eval();probs=[];saved=[];times=[];folder=out/f"{record['name']}_{record['seed']}";folder.mkdir();torch.cuda.reset_peak_memory_stats()
        for row in rows:
            x=inputs(row)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize();start=time.perf_counter();prob=model(**x).float().sigmoid()[0];torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            q=np.minimum(prob.cpu().numpy()*256,255).astype(np.uint8);path=folder/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,probability_u8=q);saved.append(dict(sample_id=row['sample_id'],path=str(path),sha256=file_hash(path)));probs.append(q.astype(np.float32)/256)
        th=ck['selection']['threshold'];result=summarize(rows,labels,probs,th);results.append(dict(name=record['name'],seed=record['seed'],result=result,supported=summarize(rows,labels,probs,th,(1,2)),vehicle=summarize(rows,labels,probs,th,(1,)),pedestrian=summarize(rows,labels,probs,th,(2,)),saved=saved,checkpoint=record['checkpoint'],checkpoint_sha256=record['sha256'],latency_ms=dict(samples=times,p50=float(np.median(times[1:])),p95=float(np.quantile(times[1:],.95)),max=float(max(times[1:]))),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20));print(record['name'],record['seed'],result['metrics'],flush=True)
    atomic_json(dest,dict(kind='actual_RTX4090_RGB_only_pixel_evaluation',split=a.split,records=results,index_sha256=file_hash(Path(a.root,'index.json')),source_sha256=file_hash(__file__),limits='AllthresholdslockedtrainingVal. Perclasssamebinaryhead notmulticlassclassifier; otherTargetclassesignored. CompareCOCOonsupportedvehicle/pedmask; conesunsupportedCOCO. Testno modelreselection.'))

if __name__=='__main__':main()
