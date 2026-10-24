"""Build old MMDetection3D extensions with isolated CUDA11.8 for Ada sm89."""
import argparse
from tools.m8a_p1_common import ROOT,run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
    toolkit=ROOT/'cuda-11.8'
    for header in ['cuda_runtime.h','cublas_v2.h','cusparse.h','cusolverDn.h']:
        assert (toolkit/'include'/header).exists(),header
    run('mmdet3d_build_'+args.attempt,[str(ROOT/'env/bin/python'),'setup.py','build_ext','--inplace'],
        cwd=ROOT/'sources/mmdetection3d',timeout=1200,
        extra_env={'CUDA_HOME':str(toolkit),'PATH':str(toolkit/'bin')+':'+str(ROOT/'env/bin')+':/usr/bin:/bin',
                   'TORCH_CUDA_ARCH_LIST':'8.9','FORCE_CUDA':'1','LD_LIBRARY_PATH':str(ROOT/'env/lib/python3.8/site-packages/torch/lib')})
if __name__=='__main__':main()
