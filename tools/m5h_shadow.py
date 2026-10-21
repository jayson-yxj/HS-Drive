"""Frozen RGB-only paired shadow. Oracle interventions are offline diagnostics only."""
import json,time
from pathlib import Path
import torch,numpy as np
from tools.m5g_common import load_pixel
from tools.m5h_freeze import verify
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5f_lift import lift
from diagnostics.m5f_metrics import histogram,metrics
from diagnostics.m5c_localization import matching
from diagnostics.m5d_metrics import voxel_3d_matching
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

TEST='/root/gpufree-data/hs-drive-data/processed/m5h_test_pixels'
def main():
    out=Path('reports/m5h_shadow_comparison.json')
    if out.exists():raise FileExistsError(out)
    frozen=verify();prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path(TEST,'index.json').read_text())['records'];pix=json.loads(Path('reports/m5h_pixel_test.json').read_text())['records'];teacher=json.loads(Path('reports/m5h_teacher_test.json').read_text())['records'][0];net=encoder_load();olddepth={k:v.detach().clone()for k,v in net.depth_head.state_dict().items()};art=Path('runs/m5h_shadow');art.mkdir(exist_ok=False);records=[]
    def scores(record,row):
        r=next(x for x in record['saved']if x['sample_id']==row['sample_id']);assert file_hash(r['path'])==r['sha256'];return torch.from_numpy(np.load(r['path'])['probability_u8'].astype(np.float32)/256).cuda()[None,:,4::8,4::8]
    for seed in [101,102]:
        choice=frozen['choices'][str(seed)];old=next(r for r in pix if r['name']=='M5G_fpn_bce'and r['seed']==seed);new=next(r for r in pix if r['name']==choice['pixel']and r['seed']==seed);distill=next(r for r in pix if r['name']=='distill'and r['seed']==seed);newdepth=torch.load(choice['depth_checkpoint'],weights_only=True)['depth']if choice['depth_checkpoint']else olddepth
        for row in rows:
            x=inputs(row);d=pixel_label(row);truth,known,actors=reference(d,x,row,row['camera_names']);supported=dict(d,target=d['target']&np.isin(d['classes'],[1,2]),valid=d['valid']&(~d['target']|np.isin(d['classes'],[1,2])));st,sv,sa=reference(supported,x,row,row['camera_names']);net.depth_head.load_state_dict(olddepth);c=cache_p8(net,x);cf=conditioned(net,c);net.depth_head.load_state_dict(newdepth)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pr=net.depth_head(cf.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2)
            po=scores(old,row);pn=scores(new,row);pd=scores(distill,row);pt=scores(teacher,row);gt=torch.from_numpy(d['target'].astype(np.float32)).cuda()[None,:,4::8,4::8];valid=torch.from_numpy(d['valid']).cuda()[None,:,4::8,4::8];dep=torch.from_numpy(d['depth']).cuda()[None,:,4::8,4::8]
            modes=[('oldpixel_olddepth',po,c['probs'],None,None),('newpixel_olddepth',pn,c['probs'],None,None),('distill_olddepth',pd,c['probs'],None,None),('oldpixel_newdepth',po,pr,None,None),('newpixel_newdepth',pn,pr,None,None),('oracle2d_newdepth',gt,pr,None,valid),('oracle2d_oracledepth',gt,pr,dep,valid),('teacher2d_oracledepth',pt,pr,dep,valid)]
            for name,fg,prob,depth,mask in modes:
                with torch.no_grad():e=lift(net.splat,fg,prob,c['K'],c['T'],c['image_hw'],depth=depth,valid=mask)
                score=e['event_score'][0].cpu().numpy();pred=score>=int(.2*256)/256;path=art/f"{seed}_{name}_{row['sample_id'].replace('/','_')}.npz";np.savez_compressed(path,score=score,target=truth,valid=known);records.append(dict(seed=seed,mode=name,group=row['group'],configuration=row['configuration'],sample_id=row['sample_id'],metrics=metrics(histogram(score,truth,known),.2),supported_metrics=metrics(histogram(score,st,sv),.2),supported_matching=matching(pred&sv,st&sv),histogram=histogram(score,truth,known).tolist(),matching=matching(pred&known,truth&known),voxel3d=voxel_3d_matching(pred&known,truth&known),unknown_high_fraction=float(pred[~known].mean()),actors=[dict(actor_id=a['actor_id'],type_id=a['type_id'],distance_m=a['distance_m'],visible_cells=int(a['mask'].sum()),matching=matching(pred&known,a['mask']))for a in actors],path=str(path),sha256=file_hash(path),control_sent=False,oracle_diagnostic=mask is not None,depth_validation_eligible=choice['depth_validation_eligible']))
            print(seed,row['sample_id'],flush=True)
    pairs=[]
    for seed in [101,102]:
        for name in sorted({r['mode']for r in records}):
            for group in sorted({r['group']for r in rows}):
                rr=sorted([r for r in records if r['seed']==seed and r['mode']==name and r['group']==group],key=lambda r:r['configuration']);a=np.load(rr[0]['path']);z=np.load(rr[-1]['path'])
                for r in rr[1:-1]:
                    d=np.load(r['path']);m=d['target']&d['valid'];n=int(m.sum());mz=m&z['valid']&~z['target'];nr=int(mz.sum());pairs.append(dict(seed=seed,mode=name,group=group,configuration=r['configuration'],n=n,removed_known_n=nr,mean_present=float(d['score'][m].mean())if n else None,mean_removed=float(z['score'][m].mean())if n else None,removed_event_residue=float((z['score'][m]>=int(.2*256)/256).mean())if n else None,confirmed_removed_fpr=float((z['score'][mz]>=int(.2*256)/256).mean())if nr else None))
    atomic_json(out,dict(kind='actual_RTX4090_recorded_RGB_offline_shadow',records=records,pairs=pairs,source_sha256=file_hash(__file__),limits='No control. FormalRGBbranches noGTmask. Teacherdoesnot supportcones; actorbytype separate. SharedP8 UV/K/T/BEV/HeightSplatthreshold. OracleDepth reference only. Currentbackground notfreevolume. Unknown residue notsafeabsence. Depth qualifiedonlyonVal untiltestgate separatelyevaluated.'))
if __name__=='__main__':main()
