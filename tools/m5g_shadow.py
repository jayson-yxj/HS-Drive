"""Recorded RGB-only joint comparison. Oracle interventions diagnostic, never controls."""
import json,time
from pathlib import Path
import torch,numpy as np
from tools.m5g_common import load_pixel
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference
from tools.m5a_learning import encoder_load
from models.m5f_pixel import ForegroundModel
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5f_lift import lift
from diagnostics.m5f_metrics import histogram,metrics
from diagnostics.m5c_localization import matching
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    out=Path('reports/m5g_shadow_comparison.json')
    if out.exists():raise FileExistsError(out)
    prepare_cuda();torch.set_num_threads(2);root=Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_pixels');rows=json.loads((root/'index.json').read_text())['records'];frozen=json.loads(Path('reports/m5g_pretest_freeze.json').read_text());pix=json.loads(Path('reports/m5g_pixel_test.json').read_text())['records'];net=encoder_load();olddepth={k:v.detach().clone()for k,v in net.depth_head.state_dict().items()};art=Path('runs/m5g_shadow');art.mkdir(exist_ok=False);records=[];timing=[]
    for seed in [101,102]:
        choice=frozen['choices'][str(seed)];old=next(r for r in pix if r['name']=='M5F_P8_fpn'and r['seed']==seed);new=next(r for r in pix if r['name']==choice['pixel']and r['seed']==seed);newdepth=torch.load(choice['depth_checkpoint'],weights_only=True)['depth'];pn,head,ck=load_pixel(dict(checkpoint=choice['pixel_checkpoint'],sha256=choice['pixel_sha256']));model=ForegroundModel(pn.encoder,head,8).eval();torch.cuda.reset_peak_memory_stats();times=[]
        for row in rows:
            x=inputs(row);d=pixel_label(row);truth,known,actors=reference(d,x,row,row['camera_names']);net.depth_head.load_state_dict(olddepth)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize();start=time.perf_counter();pnew=model(**x).float().sigmoid();c=cache_p8(net,x);cf=conditioned(net,c);net.depth_head.load_state_dict(newdepth);pr=net.depth_head(cf.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2);lifted=lift(net.splat,pnew[:,:,4::8,4::8],pr,c['K'],c['T'],c['image_hw']);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            oldrow=next(r for r in old['saved']if r['sample_id']==row['sample_id']);po=torch.from_numpy(np.load(oldrow['path'])['probability_u8'].astype(np.float32)/256).cuda()[None,:,4::8,4::8];gt=torch.from_numpy(d['target'].astype(np.float32)).cuda()[None,:,4::8,4::8];valid=torch.from_numpy(d['valid']).cuda()[None,:,4::8,4::8];dep=torch.from_numpy(d['depth']).cuda()[None,:,4::8,4::8]
            for name,fg,prob,depth,mask in [('oldpixel_olddepth',po,c['probs'],None,None),('newpixel_olddepth',pnew[:,:,4::8,4::8],c['probs'],None,None),('oracle2d_newdepth',gt,pr,None,valid),('newpixel_newdepth',pnew[:,:,4::8,4::8],pr,None,None),('oracle2d_oracledepth',gt,pr,dep,valid)]:
                with torch.no_grad():e=lift(net.splat,fg,prob,c['K'],c['T'],c['image_hw'],depth=depth,valid=mask)
                score=e['event_score'][0].cpu().numpy();pred=score>=int(.2*256)/256;path=art/f"{seed}_{name}_{row['sample_id'].replace('/','_')}.npz";np.savez_compressed(path,score=score,target=truth,valid=known);records.append(dict(seed=seed,mode=name,group=row['group'],configuration=row['configuration'],sample_id=row['sample_id'],threshold=.2,metrics=metrics(histogram(score,truth,known),.2),histogram=histogram(score,truth,known).tolist(),matching=matching(pred&known,truth&known),unknown_high_fraction=float(pred[~known].mean()),actors=[dict(actor_id=a['actor_id'],type_id=a['type_id'],distance_m=a['distance_m'],visible_cells=int(a['mask'].sum()),matching=matching(pred&known,a['mask']))for a in actors],path=str(path),sha256=file_hash(path),control_sent=False,oracle_diagnostic=mask is not None))
        timing.append(dict(seed=seed,times_ms=times,p50=float(np.median(times[1:])),p95=float(np.quantile(times[1:],.95)),max=float(max(times[1:])),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    pairs=[]
    for seed in [101,102]:
        for name in sorted({r['mode']for r in records}):
            for group in sorted({r['group']for r in rows}):
                rr=sorted([r for r in records if r['seed']==seed and r['mode']==name and r['group']==group],key=lambda r:r['configuration']);a=np.load(rr[0]['path']);z=np.load(rr[-1]['path'])
                for r in rr[1:-1]:
                    d=np.load(r['path']);mask=d['target']&d['valid'];n=int(mask.sum());pairs.append(dict(seed=seed,mode=name,group=group,configuration=r['configuration'],n=n,mean_present=float(d['score'][mask].mean())if n else None,mean_empty=float(a['score'][mask].mean())if n else None,mean_removed=float(z['score'][mask].mean())if n else None,empty_event_residue=float((a['score'][mask]>=int(.2*256)/256).mean())if n else None,removed_event_residue=float((z['score'][mask]>=int(.2*256)/256).mean())if n else None))
    atomic_json(out,dict(kind='actual_RTX4090_recordedRGB_offline_joint_shadow',records=records,pairs=pairs,timing=timing,source_sha256=file_hash(__file__),limits='No vehiclecontrol. Learnedpaths noGTMask. SameP8 originalfrozenDepthRGBfeatures acrossdeptharms, newpixel independentcopy. Oracle2DGT onlydiagnostic. Knownbackgroundcurrentvisiblefirstsurfaces not freevolume. Emptyatoldtargetcells eventresidue notcertifiedabsence. Timingincludesextraolddepthcompute andweightcopy, dedicatedlatencyseparate.'))

if __name__=='__main__':main()
