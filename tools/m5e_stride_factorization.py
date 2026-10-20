"""Fixed RGB interventions across real P8 and S16; no inference GT or new selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load,stack
from tools.m5e_common import ROOT,sidecars,pairs
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5d_stride8 import cache_p8
from models.m5e_paths import factorize,geometry
from models.m5c_obstacle_head import VisibleObstacleHead
from diagnostics.m5d_metrics import witness_metrics,aggregate_witness
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    dest=Path('reports/m5e_stride8_factorization.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);ds=VisualEvidenceDataset(ROOT,'val',supervision=False);lookup={r['sample_id']:i for i,r in enumerate(ds.rows)};sup=sidecars(ROOT,ds.rows,'m5e_supervision.json');ref=sidecars(ROOT,ds.rows,'m5e_surface_index.json');pair=pairs(ROOT,'val');net=encoder_load();inputs=[stack([ds[i]],'inputs') for i in range(len(ds))];cc16=[cache_rgb(net,i) for i in inputs];cc8=[cache_p8(net,i) for i in inputs];records=[]
    for seed in [81,82]:
        for trained in ['C_stable','S8_stable']:
            ckpath=f'runs/m5d_learning/m5d_{trained}_{seed}.pt';ck=torch.load(ckpath,weights_only=True);net=encoder_load();net.context_head.load_state_dict(ck['context']);head=VisibleObstacleHead().cuda().eval();head.load_state_dict(ck['head']);th=ck['selection']['threshold']
            for mode in ['s16','p8_s16depth','p8','p8_down16']:
                scores=[];resources=[];geometric=[];frames=[]
                for i,(row,a,b,inp) in enumerate(zip(ds.rows,cc16,cc8,inputs)):
                    c=factorize(a,b,mode);stride=8 if mode in ['p8','p8_s16depth'] else 16
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):score=head(cached_forward(net,c))['visible_obstacle'][0].float().sigmoid().cpu().numpy()
                    scores.append(score);d=ref[i];actor=d['actor_masks'].any(0)&d['valid'];ground=d['valid']&~d['target'];frames.append(dict(group=row['group'],configuration=row['configuration'],actor_n=int(actor.sum()),actor_hits=int(((score>th)&actor).sum()),ground_n=int(ground.sum()),ground_fp=int(((score>th)&ground).sum()),unknown_n=int((~d['valid']).sum()),unknown_high=int(((score>th)&~d['valid']).sum())))
                    depth=torch.from_numpy(sup[i][f'depth{stride}'][None]).cuda();m=torch.from_numpy((sup[i][f'valid{stride}']&sup[i][f'actor{stride}'])[None]).cuda();g=geometry(net,c,depth);m &= g['inside'];mass=(c['probs']*g['same_layer']*(g['xy']<=1)*g['bin_inside']).sum(2)
                    geometric.append(dict(group=row['group'],configuration=row['configuration'],feature_rays=int(m.sum()),gt_1m_mass_sum=float(mass[m].sum())))
                    if row['configuration'] in [0,3,6]:
                        times=[];st=[];bt=[];torch.cuda.reset_peak_memory_stats()
                        for rep in range(8):
                            torch.cuda.synchronize();start=time.perf_counter()
                            a1=cache_rgb(net,inp);b1=cache_p8(net,inp) if mode!='s16' else None;c1=factorize(a1,b1,mode)
                            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                                f=c1['features'];batch,n,chan,h,w=f.shape;ctx=net.context_head(f.flatten(0,1)).reshape(batch,n,64,h,w);torch.cuda.synchronize();t1=time.perf_counter();height,mass1=net.splat(ctx,c1['probs'],c1['K'],c1['T'],c1['image_hw']);torch.cuda.synchronize();t2=time.perf_counter();bev=net.bev_encoder(torch.cat((height.flatten(1,2),torch.log1p(mass1)),1));head(bev);torch.cuda.synchronize();end=time.perf_counter()
                            if rep>1:times.append((end-start)*1000);st.append((t2-t1)*1000);bt.append((end-t2)*1000)
                        resources.append(dict(group=row['group'],configuration=row['configuration'],full_forward_ms=times,splat_ms=st,bev_and_head_ms=bt,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,splat_points=c1['probs'].numel()))
                event=[dict(group=r['group'],witness=witness_metrics(scores[lookup[r['a']]],scores[lookup[r['b']]],p,th)) for r,p in pair];records.append(dict(trained=trained,seed=seed,mode=mode,threshold=th,checkpoint_sha256=file_hash(ckpath),frames=frames,geometry=geometric,resources=resources,witness=aggregate_witness(event)));print(trained,seed,mode,flush=True)
    atomic_json(dest,dict(kind='actual_RTX4090_frozen_M5D_heads_new_validation_factorized_interventions',records=records,source_sha256=file_hash(__file__),limits='RealP8 visualfeatures fromlayer2 not interpolatedS16context. p8_s16depth upsamples probabilities with renormalization; p8_down16 poolsP8 features but uses S16depth. Changedraycenters/probabilityinterpolation confound attribution. Frozenhead distribution shift not trainableupperbound. M5D C/S8 previously trained matched budgets; no newhighrestraining. Fullforward diagnostics p8 modes currently compute bothS16andP8 encoders: explicit redundant diagnostic cost, not optimizeddeploylatency. AllinputsRGBonly; GT solely metrics.'))


if __name__=='__main__':main()
