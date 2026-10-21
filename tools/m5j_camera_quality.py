"""Read-only historical camera/error stratification; never used to retrain/reselect."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5j_common import rows,write
from tools.m5f_common import inputs,pixel_label
from tools.m5i_data import load
from diagnostics.m5i_metrics import enhanced
from tools.m5g_evaluation import summarize
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='historical');a=p.parse_args();records=[]
    if a.split=='historical':study=json.loads(Path('reports/m5i_pixel_test.json').read_text())['records']
    else:
        from tools.m5j_freeze import verify
        verify();prepare_cuda();torch.set_num_threads(2);rr=rows('test');labels=[pixel_label(r)for r in rr];cfg=json.loads(Path('configs/m5j_models.json').read_text());dest=Path('runs/m5j_pixel_test');dest.mkdir(exist_ok=False);study=[];val=json.loads(Path('reports/m5i_pixel_val.json').read_text())['records']
        for r in cfg['pixel']:
            model,ck=load(r);probs=[];saved=[];priorname=r['name']if r['name']!='legacy'else'M5G_fpn_bce';locked=next(v for v in val if v['name']==priorname and v['seed']==r['seed'])
            for row in rr:
                x=inputs(row)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):q=np.minimum(model(**x).float().sigmoid()[0].cpu().numpy()*256,255).astype(np.uint8)
                path=dest/(r['name']+str(r['seed'])+'_'+row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,probability_u8=q);probs.append(q.astype(np.float32)/256);saved.append(dict(sample_id=row['sample_id'],path=str(path),sha256=file_hash(path)))
            study.append(dict(name=r['name'],seed=r['seed'],result=enhanced(rr,labels,probs,locked['result']['threshold'],locked_low=locked['result']['low_fpr_validation']),classes={key:summarize(rr,labels,probs,locked['result']['threshold'],cl)for key,cl in [('vehicle',(1,)),('pedestrian',(2,)),('cone',(3,))]},saved=saved));print(r['name'],r['seed'],study[-1]['result']['metrics'],flush=True)
    for r in study:
        if r['name']not in ['mobile_gt','legacy','M5G_fpn_bce']:continue
        q=r['result'];sizes=[]
        for lo,hi in [(3,32),(33,128),(129,10**9)]:
            aa=[v for v in q['actors']if lo<=v['pixels']<=hi];sizes.append(dict(area=[lo,hi],actor_views=len(aa),detected=sum(v['detected']for v in aa),tp_pixels=sum(v['hits']for v in aa),fn_pixels=sum(v['pixels']-v['hits']for v in aa),fp='Cannot assign background FP uniquelytoGTActor; camera counts/components separately'))
        records.append(dict(name=r['name'],seed=r['seed'],metrics=q['metrics'],by_scene=q['by_scene'],strata=q['strata'],actors=q['actors'],sizes=sizes,pairs=q['pairs'],false_components=q['false_detection_components'],unknown=q['unknown_response'],classes={k:v['metrics']for k,v in r['classes'].items()},saved=r['saved']))
    write(f'reports/m5j_camera_quality_{a.split}.json',dict(kind='offline_historical_regression_no_selection'if a.split=='historical'else'actual_GPU_frozen_new_location_pixel_evaluation',records=records,source_sha256=file_hash(__file__),limits='Binaryforeground classconditional masks notmulticlassclassification. Actor-view correlated; <=32pixel support maybesmall; no oldtest training.'))
if __name__=='__main__':main()
