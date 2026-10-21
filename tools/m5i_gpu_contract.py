"""CUDA inference isolation; feature-gradient behavior has separate synthetic tests."""
import json
from pathlib import Path
import torch
from models.m5i_visual import MobilePixel,align_teacher
from tools.m5i_data import rows,load
from tools.m5f_common import inputs,pixel_label
from training.m5i_transfer import feature_kd,feature_regions
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    out=Path('reports/m5i_gpu_contract.json')
    if out.exists():raise FileExistsError(out)
    prepare_cuda();torch.set_num_threads(2);row=rows()[0];x=inputs(row);d=pixel_label(row);study=json.loads(Path('reports/m5i_distillation_comparison.json').read_text());records=[]
    extra=Path('reports/m5i_distillation_comparison_feature.json')
    if extra.exists():study['records']+=json.loads(extra.read_text())['records']
    for record in study['records']:
        model,ck=load(record)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=model(**x).clone()
        d['target'][...]=False;d['depth'][...]=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):b=model(**x).clone()
        assert torch.equal(a,b);reject=[]
        for k in ['depth','teacher_feature','actor','target']:
            try:model(**dict(x,**{k:torch.zeros(1,device='cuda')}))
            except(ValueError,TypeError):reject.append(k)
        assert len(reject)==4;records.append(dict(name=record['name'],seed=record['seed'],gt_mutation_identical=True,rejected=reject,output_shape=list(a.shape)))
    original=encoder_load();cache=cache_p8(original,x);assert cache['probs'].dtype==torch.float32
    atomic_json(out,dict(kind='actual_RTX4090_RGB_only_contract_check',records=records,originalDepthProbFP32=True,source_sha256=file_hash(__file__),limits='No testGT or teacher feature is accepted byinference; featureadapter onlytraincheckpoint.'))
if __name__=='__main__':main()
