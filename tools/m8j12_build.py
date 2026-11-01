"""Build only required CUDA inference extensions in the isolated official runtime."""
from pathlib import Path
import subprocess,os,json,hashlib,argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='first');args=parser.parse_args()
    c=json.loads(Path('configs/m8j12_protocol.json').read_text());rt=Path(c['runtime_root']);py=str(rt/'env/bin/python')
    cuda='/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/cuda-11.8'
    e=dict(os.environ,CUDA_HOME=cuda,PATH=str(rt/'env/bin')+':'+cuda+'/bin:'+os.environ['PATH'],LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+cuda+'/lib',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',MMCV_WITH_OPS='1',FORCE_CUDA='1')
    commands=[([py,'-m','pip','install','--no-deps','--no-build-isolation',str(rt/'deps/mmcv-full-1.7.1')],str(rt)),([py,'setup.py','build_ext','--inplace'],str(rt/'runtime-source/projects/mmdet3d_plugin/ops'))]
    for i,(cmd,cwd) in enumerate(commands):
        log=rt/('cuda_build_'+args.attempt+'_'+str(i)+'.txt')
        with log.open('x') as f:
            r=subprocess.run(cmd,cwd=cwd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
        record=dict(command=cmd,cwd=cwd,exit_code=r.returncode,log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),CUDA_ARCH='8.9',torch='2.0.1+cu118')
        Path('reports/m8j12_sources/cuda_build_'+args.attempt+'_'+str(i)+'.json').write_text(json.dumps(record,indent=2)+'\n')
        if r.returncode:raise RuntimeError('CudaBuildFailed:'+str(i))
    print('Required extensions built')
if __name__=='__main__':main()
