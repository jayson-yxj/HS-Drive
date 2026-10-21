"""Read-only original M4 S16 near-task regression of fixed Replay candidates."""
import json
from pathlib import Path
import torch
from tools.m5a_learning import encoder_load
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    p=Path('reports/m5i_depth_original_task.json')
    if p.exists():raise FileExistsError(p)
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();base={k:v.clone()for k,v in net.depth_head.state_dict().items()};models=[dict(name='original',seed=0,weights=base)];old=json.loads(Path('reports/m5h_depth_preservation.json').read_text())
    for r in old['records']:
        if r['name']=='replay':models.append(dict(name='replay',seed=r['seed'],weights=torch.load(r['selected']['checkpoint'],weights_only=True)['depth'],sha256=r['selected']['sha256']))
    rows=[r for r in json.loads(Path('runs/m5h_replay/index.json').read_text())['records']if r['split']=='val'and r['stride']==16];records=[]
    for r in rows:
        d={k:v.cuda()for k,v in torch.load(r['path'],weights_only=True).items()}
        for m in models:
            net.depth_head.load_state_dict(m['weights'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pr=net.depth_head(d['rgb_conditioned']).float().softmax(1)
            exp=(pr*net.splat.depth_centers[None,:,None,None]).sum(1);delta=exp-d['depth'];mask=d['valid'];records.append(dict(name=m['name'],seed=m['seed'],sample_id=r['sample_id'],n=int(mask.sum()),mae=float(delta[mask].abs().mean()),bias=float(delta[mask].mean())))
    atomic_json(p,dict(kind='actual_GPU_original_S16_M4_validation_near_replay_check',records=records,source_sha256=file_hash(__file__),limits='Cachecontains only1.5-10m reliableoriginalvalidationmask; no newtraining; originalTask otherlossesnotrequalified.'))
if __name__=='__main__':main()
