"""No private data/model: minimal official DCNv2 gradient repeatability probe."""
import json
from pathlib import Path
import os
import torch
from mmcv.ops import ModulatedDeformConv2dPack
from tools.m8j5c_numeric import metric


def main():
    torch.manual_seed(850541);torch.cuda.manual_seed_all(850541)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(False)
    model=ModulatedDeformConv2dPack(512,512,3,padding=1,bias=False).cuda().eval()
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    inp=torch.randn(6,512,2,3,device='cuda')*.1
    upstream=torch.randn(6,512,2,3,device='cuda')*.03
    results={}
    for dtype in (torch.float32,torch.float16):
        runs=[]
        for i in range(3):
            model.load_state_dict(state);model.zero_grad(set_to_none=True)
            x=inp.to(dtype).detach().clone().requires_grad_()
            with torch.autocast('cuda',dtype=torch.float16,enabled=dtype==torch.float16):
                y=model(x)
            y.backward(upstream.to(y.dtype))
            runs.append(dict(input=x.grad.detach().float().cpu(),
                weight=model.weight.grad.detach().float().cpu(),
                offset_weight=model.conv_offset.weight.grad.detach().float().cpu()))
        results[str(dtype)]=[{n:metric(runs[0][n],runs[i][n]) for n in runs[0]} for i in (1,2)]
    out=dict(seed=850541,shape=[6,512,2,3],evidence='ActualCUDA_MinimalSyntheticOperator_NoDataset',
        module='OfficialMMCV_ModulatedDeformConv2dPack',model_mode='eval',scaler='NotApplicable_NoScaling',
        input_parameters_upstream_fixed=True,gradient_cleared_each_run=True,optimizer_steps=0,results=results)
    with (Path(os.environ['M8J5C_OUTPUT'])/'dcn_minimal.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))


if __name__=='__main__':main()
