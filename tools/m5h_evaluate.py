"""Same reliable domains for teacher/student, validation workpoints then frozen test."""
import argparse,json,time
from pathlib import Path
import torch,numpy as np
from tools.m5h_data import rows,ROOT
from tools.m5g_common import load_pixel
from tools.m5f_common import inputs,pixel_label
from models.m5f_pixel import ForegroundModel
from diagnostics.m5h_metrics import enhanced
from tools.m5g_evaluation import summarize
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--split',default='val');p.add_argument('--teacher',action='store_true');p.add_argument('--suffix',default='');a=p.parse_args();kind='teacher'if a.teacher else'pixel';dest=Path(f'reports/m5h_{kind}_{a.split}{a.suffix}.json');out=Path(f'runs/m5h_{kind}_eval_{a.split}{a.suffix}');out.mkdir(exist_ok=False)
    if dest.exists():raise FileExistsError(dest)
    if a.split=='test':
        from tools.m5h_freeze import verify
        verify()
    prepare_cuda();torch.set_num_threads(2);rr=[r for r in rows(a.root)if r['split']==a.split];labels=[pixel_label(r)for r in rr];locked=json.loads(Path(f'reports/m5h_{kind}_val.json').read_text())if a.split!='val'else None
    if a.teacher:study=[dict(name='COCO_teacher',seed=0)]
    else:
        study=json.loads(Path('reports/m5h_distillation_ablation.json').read_text())['records'];old=json.loads(Path('reports/m5g_backbone_ablation.json').read_text())['records'];study+=[dict(r,name='M5G_'+r['name'])for r in old if r['name']=='fpn_bce']
    results=[]
    for record in study:
        probs=[];saved=[];times=[];name=record['name'];seed=record['seed'];folder=out/f'{name}_{seed}';folder.mkdir();torch.cuda.reset_peak_memory_stats()
        if not a.teacher:net,head,ck=load_pixel(record);model=ForegroundModel(net.encoder,head,8).eval()
        else:
            cache=Path('runs/m5h_teacher_'+('test'if a.split=='test'else'trainval'));idx={r['sample_id']:r for r in json.loads((cache/'index.json').read_text())['records']}
        for row in rr:
            if a.teacher:
                rec=idx[row['sample_id']];assert file_hash(rec['path'])==rec['sha256'];q=np.load(rec['path'])['score'].max(0)
            else:
                x=inputs(row)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    torch.cuda.synchronize();start=time.perf_counter();prob=model(**x).float().sigmoid()[0];torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                q=np.minimum(prob.cpu().numpy()*256,255).astype(np.uint8)
            path=folder/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,probability_u8=q);saved.append(dict(sample_id=row['sample_id'],path=str(path),sha256=file_hash(path)));probs.append(q.astype(np.float32)/256)
        lockedrec=next(r for r in locked['records']if r['name']==name and r['seed']==seed)if locked else None;th=(lockedrec['result']['threshold']if locked else None)if a.teacher else ck['selection']['threshold'];classes=(1,2)if a.teacher else(1,2,3)
        result=enhanced(rr,labels,probs,th,classes,lockedrec['result']['low_fpr_validation']if locked else None);supported=enhanced(rr,labels,probs,th,(1,2),lockedrec['supported']['low_fpr_validation']if locked else None);results.append(dict(name=name,seed=seed,result=result,supported=supported,classes={key:summarize(rr,labels,probs,result['threshold'],cl)for key,cl in [('vehicle',(1,)),('pedestrian',(2,)),('cone',(3,))]if not(a.teacher and key=='cone')},saved=saved,checkpoint=record.get('checkpoint'),checkpoint_sha256=record.get('sha256'),times_ms=times,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20));print(name,seed,result['metrics'],flush=True)
    atomic_json(dest,dict(kind='actual_GPU_cachedteacher_or_RGB_student_evaluation',split=a.split,records=results,source_sha256=file_hash(__file__),index_sha256=file_hash(Path(a.root,'index.json'))))
if __name__=='__main__':main()
