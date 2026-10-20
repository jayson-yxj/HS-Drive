"""Actual CUDA/BF16 forward-backward and scatter smoke test (no simulator)."""
import argparse,json,subprocess
import torch
import torchvision
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    checks={'cuda_available':torch.cuda.is_available()}
    result={'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_runtime':torch.version.cuda,'checks':checks}
    if checks['cuda_available']:
        from training.runtime import prepare_cuda
        result['linalg_absolute_preload']=prepare_cuda()
        checks['linalg_inverse']=bool(torch.allclose(torch.linalg.inv(torch.eye(3,device='cuda')),torch.eye(3,device='cuda')))
        torch.cuda.reset_peak_memory_stats();torch.manual_seed(7)
        result.update(device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
            driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
        checks['bf16_supported']=torch.cuda.is_bf16_supported()
        x=torch.randn(2,3,128,128,device='cuda',requires_grad=True)
        layer=torch.nn.Conv2d(3,16,3,padding=1).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):y=layer(x)
        y.float().square().mean().backward()
        checks['bf16_conv_backward']=bool(y.dtype==torch.bfloat16 and torch.isfinite(layer.weight.grad).all())
        source=torch.randn(100,4,device='cuda',requires_grad=True);idx=torch.arange(100,device='cuda')%10
        out=source.new_zeros(10,4).index_add_(0,idx,source)
        out.square().sum().backward();torch.cuda.synchronize()
        checks['scatter_backward']=bool(torch.isfinite(source.grad).all())
        result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
    result['passed']=all(checks.values());atomic_json(a.output,result);print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
