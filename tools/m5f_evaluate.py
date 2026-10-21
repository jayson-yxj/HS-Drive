"""Actual singleframe pixel evaluation; paired masks only offline, no Gold Absence."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from models.m5f_pixel import ForegroundModel,PixelHead
from tools.m5a_learning import encoder_load
from tools.m5f_common import inputs,pixel_label,checkpoint
from diagnostics.m5f_metrics import histogram,metrics
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from tools.m5f_dataset import ROOT


def evaluate(record,rows,art):
    ck=checkpoint(record);net=encoder_load();net.encoder.lateral.load_state_dict(ck['lateral']);net.encoder.output.load_state_dict(ck['output']);head=PixelHead().cuda().eval();head.load_state_dict(ck['head']);model=ForegroundModel(net.encoder,head,ck['stride']).eval();threshold=ck['selection']['threshold'];cut=int(threshold*256);hist={};frames=[];scores={};labels={};times=[];paths=[]
    torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        inp=inputs(row)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.cuda.synchronize();start=time.perf_counter();p=model(**inp).float().sigmoid()[0];torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        p=p.cpu().numpy();quant=np.minimum((p*256).astype(np.uint16),255).astype(np.uint8);path=art/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,probability_u8=quant);paths.append(dict(sample_id=row['sample_id'],path=str(path),sha256=file_hash(path)));scores[row['sample_id']]=quant;d=pixel_label(row);labels[row['sample_id']]=d;pred=quant>=cut;actors=[]
        for c,name in enumerate(row['camera_names']):
            masks={'all':d['valid'][c],'ground':d['valid'][c]&~d['target'][c]}
            masks.update({f'distance_{lo}_{hi}':d['valid'][c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi) for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]})
            for k,m in masks.items():
                h=histogram(p[c],d['target'][c],m)
                for key in [f'{row["group"]}/{k}',f'camera:{name}/{k}',f'weather:{row["weather"]}/{k}',f'pooled/{k}']:hist[key]=hist.get(key,np.zeros((2,256),np.int64))+h
                if row['configuration'] in [0,15,11] and k=='all' and not d['target'].any():hist['empty']=hist.get('empty',np.zeros((2,256),np.int64))+h
            for a in row['actors'][c]['actors']:
                m=d['actor'][c]==a['local_id'];n=int(m.sum());hit=int((pred[c]&m).sum());rec=hit/n if n else None
                actors.append(dict(**a,camera=name,predicted_pixels=hit,recall=rec,detected=n>=3 and hit>=3 and rec>=.5,size_bucket='small_le32' if n<=32 else 'medium_le128' if n<=128 else 'large',mean_probability=float(p[c][m].mean()) if n else None))
        frames.append(dict(sample_id=row['sample_id'],group=row['group'],configuration=row['configuration'],actors=actors,overall=metrics(histogram(p,d['target'],d['valid']),threshold),known_fraction=float(d['valid'].mean()),ignored_high_fraction=float(pred[~d['valid']].mean())));times.append(elapsed)
    pairs=[]
    for g in sorted({r['group'] for r in rows}):
        rr=sorted([r for r in rows if r['group']==g],key=lambda x:x['configuration']);a,z=rr[0],rr[-1];pa=scores[a['sample_id']].astype(float)/256;pz=scores[z['sample_id']].astype(float)/256;da,dz=labels[a['sample_id']],labels[z['sample_id']]
        for row in rr[1:-1]:
            p=scores[row['sample_id']].astype(float)/256;d=labels[row['sample_id']]
            m=d['target']&d['valid']&da['valid']&~da['target']&(d['actor']>0);mz=m&dz['valid']&~dz['target'];n=int(m.sum());nr=int(mz.sum())
            pairs.append(dict(group=g,configuration=row['configuration'],n=n,presence_hits=int(((p>=cut/256)&m).sum()),empty_fp=int(((pa>=cut/256)&m).sum()),both_correct=int(((p>=cut/256)&(pa<cut/256)&m).sum()),mean_present=float(p[m].mean()) if n else None,mean_empty=float(pa[m].mean()) if n else None,mean_rise=float((p-pa)[m].mean()) if n else None,direction_rise=float(((p-pa)[m]>0).mean()) if n else None,removed_n=nr,mean_removed=float(pz[mz].mean()) if nr else None,removed_fp=int(((pz>=cut/256)&mz).sum()),direction_fall=float(((p-pz)[mz]>0).mean()) if nr else None,by_distance={f'{lo}_{hi}':dict(n=int((m&(d['radial']>=lo)&(d['radial']<hi)).sum()),presence_hits=int(((p>=cut/256)&m&(d['radial']>=lo)&(d['radial']<hi)).sum()),empty_fp=int(((pa>=cut/256)&m&(d['radial']>=lo)&(d['radial']<hi)).sum())) for lo,hi in [(5,10),(10,15),(15,20),(20,35)]}))
    return dict(name=record['name'],seed=record['seed'],stride=record['stride'],threshold=threshold,effective_quantized_threshold=cut/256,frames=frames,pairs=pairs,strata={k:metrics(h,threshold) for k,h in hist.items()},histograms={k:h.tolist() for k,h in hist.items() if k.endswith('/all') or k=='empty'},predictions=paths,latency_ms=dict(samples=times,p50=float(np.median(times[1:])),p95=float(np.quantile(times[1:],.95)),max=float(max(times[1:]))),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha256=record['sha256'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default=ROOT);p.add_argument('--split',default='val');a=p.parse_args();dest=Path(f'reports/m5f_{"pixel_validation" if a.split=="val" else "2d_presence_absence"}.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);rows=[r for r in json.loads(Path(a.index,'index.json').read_text())['records'] if r['split']==a.split];study=json.loads(Path('reports/m5f_image_foreground_training.json').read_text());out=[];art=Path(f'runs/m5f_{a.split}_predictions');art.mkdir(exist_ok=False)
    if a.split=='test':
        freeze=json.loads(Path('reports/m5f_pretest_decision.json').read_text());assert freeze['protocol_sha256']==file_hash('configs/m5f_protocol.json')
        assert all(file_hash(p)==h for p,h in freeze['checkpoints'].items())
    for r in study['records']:
        folder=art/f"{r['name']}_{r['seed']}";folder.mkdir();result=evaluate(r,rows,folder);out.append(result);print(r['name'],r['seed'],result['strata']['pooled/all'],flush=True)
    atomic_json(dest,dict(kind='actual_RTX4090_singleframe_RGB_pixel_evaluation',split=a.split,records=out,index_sha256=file_hash(Path(a.index,'index.json')),source_sha256=file_hash(__file__),limits='Pixelvalid currentrenderedfirstsurface not3Dabsence; instance association geometricproxy; invisibleactors retainedzero. 256binAP and floor256threshold consistentlyused. Savedquantization<1/256. Pairs offline only, model receivesoneframeRGB. No formaldriving.'))

if __name__=='__main__':main()
