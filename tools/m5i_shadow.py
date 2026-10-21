"""Frozen RGB-only architecture/depth shadow; six-second runtime unchanged."""
import json
from pathlib import Path
import torch,numpy as np
from tools.m5i_data import TEST
from tools.m5i_freeze import verify
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

def main():
    out=Path('reports/m5i_shadow_comparison.json')
    if out.exists():raise FileExistsError(out)
    freeze=verify();prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path(TEST,'index.json').read_text())['records'];pix=json.loads(Path('reports/m5i_pixel_test.json').read_text())['records'];teacher=json.loads(Path('reports/m5i_teacher_test.json').read_text())['records'][0];net=encoder_load();olddepth={k:v.clone()for k,v in net.depth_head.state_dict().items()};records=[];dest=Path('runs/m5i_shadow');dest.mkdir(exist_ok=False)
    def scores(record,row):
        r=next(r for r in record['saved']if r['sample_id']==row['sample_id']);assert file_hash(r['path'])==r['sha256'];return torch.from_numpy(np.load(r['path'])['probability_u8'].astype(np.float32)/256).cuda()[None,:,4::8,4::8]
    for seed in [101,102]:
        c=freeze['choices'][str(seed)];old=next(r for r in pix if r['name']=='M5G_fpn_bce'and r['seed']==seed);new=next(r for r in pix if r['name']==c['pixel']and r['seed']==seed);dw=torch.load(c['depth_checkpoint'],weights_only=True)['depth']
        for row in rows:
            x=inputs(row);d=pixel_label(row);truth,valid,actors=reference(d,x,row,row['camera_names']);support=dict(d,target=d['target']&np.isin(d['classes'],[1,2]),valid=d['valid']&(~d['target']|np.isin(d['classes'],[1,2])));st,sv,_=reference(support,x,row,row['camera_names']);net.depth_head.load_state_dict(olddepth);cache=cache_p8(net,x);f=conditioned(net,cache);net.depth_head.load_state_dict(dw)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pr=net.depth_head(f.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2)
            po=scores(old,row);pn=scores(new,row);pt=scores(teacher,row);gt=torch.from_numpy(d['target']).float().cuda()[None,:,4::8,4::8];v=torch.from_numpy(d['valid']).cuda()[None,:,4::8,4::8];dep=torch.from_numpy(d['depth']).cuda()[None,:,4::8,4::8]
            for mode,fg,probs,depth,mask in [('old_old',po,cache['probs'],None,None),('new_old',pn,cache['probs'],None,None),('old_replay',po,pr,None,None),('new_replay',pn,pr,None,None),('teacher_oracle',pt,pr,dep,v),('oracle_oracle',gt,pr,dep,v)]:
                with torch.no_grad():output=lift(net.splat,fg,probs,cache['K'],cache['T'],cache['image_hw'],depth=depth,valid=mask)
                score=output['event_score'][0].cpu().numpy();pred=score>=int(.2*256)/256;p=dest/f"{seed}_{mode}_{row['sample_id'].replace('/','_')}.npz";np.savez_compressed(p,score=score,target=truth,valid=valid);records.append(dict(seed=seed,mode=mode,group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],metrics=metrics(histogram(score,truth,valid),.2),matching=matching(pred&valid,truth&valid),voxel3d=voxel_3d_matching(pred&valid,truth&valid),supported_metrics=metrics(histogram(score,st,sv),.2),supported_matching=matching(pred&sv,st&sv),unknown_high_fraction=float(pred[~valid].mean()),actors=[dict(type_id=a['type_id'],distance_m=a['distance_m'],visible_cells=int(a['mask'].sum()),matching=matching(pred&valid,a['mask']))for a in actors],path=str(p),sha256=file_hash(p),pixel_joint_qualified=c['pixel_joint_qualified'],control_sent=False))
            print(seed,row['sample_id'],flush=True)
    pairs=[]
    for seed in [101,102]:
        for mode in sorted({r['mode']for r in records}):
            for group in sorted({r['group']for r in rows}):
                rr=sorted([r for r in records if r['seed']==seed and r['mode']==mode and r['group']==group],key=lambda r:r['configuration']);z=np.load(rr[-1]['path'])
                for r in rr[1:-1]:
                    d=np.load(r['path']);m=d['target']&d['valid'];known=m&z['valid']&~z['target'];n=int(m.sum());nk=int(known.sum());pairs.append(dict(seed=seed,mode=mode,group=group,configuration=r['configuration'],n=n,removed_known_n=nk,event_residue=float((z['score'][m]>=.19921875).mean())if n else None,confirmed_removed_fpr=float((z['score'][known]>=.19921875).mean())if nk else None))
    atomic_json(out,dict(kind='actual_GPU_offline_recordedRGB_shadow',records=records,pairs=pairs,source_sha256=file_hash(__file__),limits='Unqualifiedstudent diagnostic notdeployment. P8UV originalK/T/HeightSplat/.2 threshold fixed. OracleGT onlydiagnostic; formalstudentnoGT. Knownbackground notfreevolume; unknown notnegative. Coarsevoxelcenter3D notphysicalActorerror.'))
if __name__=='__main__':main()
