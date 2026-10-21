"""Actual CUDA isolation/gradient check, independent of scientific evaluation."""
import json,copy
from pathlib import Path
import torch
from models.m5f_pixel import ForegroundModel
from tools.m5g_common import load_pixel
from tools.m5f_common import inputs,pixel_label
from tools.m5h_data import rows
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    dest=Path('reports/m5h_gpu_contract.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);record=json.loads(Path('reports/m5h_distillation_ablation.json').read_text())['records'][0];net,head,ck=load_pixel(record);model=ForegroundModel(net.encoder,head,8).eval();row=next(r for r in rows()if r['split']=='train');x=inputs(row);d=pixel_label(row)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=model(**x).clone()
    for k in ['target','valid','classes','depth']:d[k][...]=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):b=model(**x).clone()
    assert torch.equal(a,b)
    rejected=[]
    for key in ['target','depth','actor','teacher_score','valid_mask']:
        try:model(**dict(x,**{key:torch.zeros(1,device='cuda')}))
        except (ValueError,TypeError):rejected.append(key)
    assert len(rejected)==5
    original=encoder_load();depth_changed=any(not torch.equal(v,original.depth_head.state_dict()[k])for k,v in net.depth_head.state_dict().items());assert not depth_changed
    fpn_changed=any(not torch.equal(v,original.encoder.output.state_dict()[k])for k,v in net.encoder.output.state_dict().items());assert fpn_changed
    c=cache_p8(original,x);assert c['probs'].dtype==torch.float32
    atomic_json(dest,dict(kind='actual_RTX4090_contract_check',gpu=torch.cuda.get_device_name(),torch=torch.__version__,gt_mutation_output_identical=True,forbidden_fields_rejected=rejected,original_depth_unchanged=True,fpn_updated=True,geometry_probability_fp32=True,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
