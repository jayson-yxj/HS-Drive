"""Actual GPU auxiliary path fidelity and supervision isolation regression."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5f_common import inputs,pixel_label,checkpoint
from tools.m5f_dataset import ROOT
from tools.m5a_learning import encoder_load
from models.m5f_pixel import backbone_levels,pyramid,ForegroundModel,PixelHead
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_gpu_contract.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);row=next(r for r in json.loads(Path(ROOT,'index.json').read_text())['records'] if r['split']=='val');inp=inputs(row);net=encoder_load();record=json.loads(Path('reports/m5f_image_foreground_training.json').read_text())['records'][0];ck=checkpoint(record);head=PixelHead().cuda();head.load_state_dict(ck['head']);model=ForegroundModel(net.encoder,head,16).eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        old=net.encoder(inp['images']).flatten(0,1);new=pyramid(net.encoder,backbone_levels(net.encoder,inp['images']),16);p=model(**inp);d=pixel_label(row);d['target'][:]=~d['target'];d['depth'][:]=np.nan;p2=model(**inp)
    rejected=[]
    for key in ['depth_gt','semantic','instance','actors','valid_mask','target']:
        try:model(**dict(inp,**{key:torch.zeros(1,device='cuda')}))
        except ValueError:rejected.append(key)
    assert torch.equal(old,new) and torch.equal(p,p2) and len(rejected)==6
    atomic_json(dest,dict(kind='actual_RTX4090_sixcamera_contract_test',gpu=torch.cuda.get_device_name(),bf16_s16_matches_original_exactly=torch.equal(old,new),mutated_supervision_does_not_change_rgb_output=torch.equal(p,p2),privileged_keys_rejected=rejected,output_shape=list(p.shape),source_sha256=file_hash(__file__),limits='Contract test, not quality or driving. Existing geometry also syntheticCPUtested; no testsubset modelselection.'))

if __name__=='__main__':main()
