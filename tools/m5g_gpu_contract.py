"""Actual GPU high-layer isolation and independent supervision contract checks."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5g_common import load_pixel
from tools.m5f_common import inputs,pixel_label
from tools.m5f_dataset import ROOT
from models.m5f_pixel import ForegroundModel
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    dest=Path('reports/m5g_gpu_contract.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path(ROOT,'index.json').read_text())['records'];x=inputs(rows[0]);record=next(r for r in json.loads(Path('reports/m5g_backbone_ablation.json').read_text())['records']if r['name']=='high_bce');net,head,ck=load_pixel(record);model=ForegroundModel(net.encoder,head,8).eval();old=encoder_load()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=model(**x);before=cache_p8(old,x)['probs'];d=pixel_label(rows[0]);d['target'][:]=~d['target'];d['depth'][:]=np.nan;q=model(**x);after=cache_p8(old,x)['probs']
    assert torch.equal(p,q)and torch.equal(before,after)
    rejected=[]
    for key in ['depth_gt','target','semantic','actors','valid_mask']:
        try:model(**dict(x,**{key:torch.zeros(1,device='cuda')}))
        except ValueError:rejected.append(key)
    assert len(rejected)==5
    oldnet=encoder_load();changed={k:not torch.equal(v,oldnet.encoder.backbone.layer3.state_dict()[k])for k,v in net.encoder.backbone.layer3.state_dict().items()}
    assert any(changed.values())
    atomic_json(dest,dict(kind='actual_RTX4090_sixcamera_contract_regression',high_layer_parameters_changed=sum(changed.values()),mutated_GT_does_not_change_inference=True,original_depth_output_unchanged=True,privileged_keys_rejected=rejected,source_sha256=file_hash(__file__),limits='Contracttests notqualityorvisualsafety.'))

if __name__=='__main__':main()
