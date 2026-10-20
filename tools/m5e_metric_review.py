"""Supplement pure-height mass and geometric support denominators, fixed validation."""
import json
from pathlib import Path
import torch
import numpy as np
from tools.m5e_common import ROOT,sidecars,load_checkpoint
from tools.m5a_learning import encoder_load,stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb
from models.m5e_paths import geometry,depth_logits
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    out=Path('reports/m5e_depth_metric_review.json')
    if out.exists():raise FileExistsError(out)
    prepare_cuda();torch.set_num_threads(2);ds=VisualEvidenceDataset(ROOT,'val',supervision=False);ss=sidecars(ROOT,ds.rows,'m5e_supervision.json');net=encoder_load();caches=[cache_rgb(net,stack([ds[i]],'inputs')) for i in range(len(ds))];records=[]
    models=[dict(name='original_depth',seed=0,checkpoint=None)]+[r for r in json.loads(Path('reports/m5e_model_ablation.json').read_text())['records'] if r['name']=='D_depth']
    for model in models:
        net=encoder_load()
        if model['checkpoint']:net.depth_head.load_state_dict(load_checkpoint(model['checkpoint'])['depth'])
        for row,c,s in zip(ds.rows,caches,ss):
            d=torch.from_numpy(s['depth16'][None]).cuda();g=geometry(net,c,d)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=depth_logits(net,c).float().softmax(2)
            same=torch.bucketize(g['points'][...,2].contiguous(),net.splat.z_edges,right=True)==torch.bucketize(g['gt'][...,2].contiguous(),net.splat.z_edges,right=True)[:,:,None]
            exp=(p*net.splat.depth_centers[None,None,:,None,None]).sum(2)
            for target in ['all_reliable','actor']:
                valid=torch.from_numpy((s['valid16'] if target=='all_reliable' else s['valid16']&s['actor16'])[None]).cuda()
                for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]:
                    m=valid&(g['rad']>=lo)&(g['rad']<hi)
                    if not m.any():continue
                    outside=m&~g['inside'];support=g['same_layer']&(g['xy']<=2)
                    records.append(dict(name=model['name'],seed=model['seed'],group=row['group'],target=target,distance=f'{lo}_{hi}',n=int(m.sum()),gt_inside=int((m&g['inside']).sum()),outside_n=int(outside.sum()),pure_correct_height_mass=float((p*same).sum(2)[m].mean()),correct_height_and_bev_mass=float((p*g['same_layer']).sum(2)[m].mean()),expected_depth_mae_m=float((exp-d).abs()[m].mean()),spatial_loss_supported_n=int((m&g['inside']&support.any(2)).sum()),outside_gt_entering_bev_conditional_mass=float((p*g['bin_inside']).sum(2)[outside].mean()) if outside.any() else None))
    atomic_json(out,dict(kind='actual_GPU_additive_metric_definition_review_fixed_validation',records=records,source_sha256=file_hash(__file__),limits='Original report correct_layer_mass denotes heightANDinsideBEV for probability rows; pureheight supplied here without changing original reports or checkpoints/selection. Outsideentering original is mean over allrayrows, here conditional denominator explicit. Geometry same rounded UV. No model selection.'))

if __name__=='__main__':main()
