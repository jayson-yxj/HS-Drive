"""Restricted CPU unpickling; separate no-network numeric-only GPU subprocess."""
import argparse
from pathlib import Path
import subprocess
import threading
import time
from tools.m8a_p1_common import ROOT,LOG,CFG,provenance,write,asset,run


def command(output, script):
    pybase=Path('/root/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu')
    cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
        '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
        '--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache',
        '--ro-bind',str(ROOT),str(ROOT),'--ro-bind',str(pybase),str(pybase),
        '--ro-bind',CFG['checkpoint'],'/input/epoch_24.pth',
        '--ro-bind',str(script.resolve()),'/gpu_probe.py','--bind',str(output),'/output',
        '--tmpfs','/tmp','--tmpfs','/proc','--ro-bind','/sys','/sys','--dev','/dev']
    for name in ['nvidia0','nvidiactl','nvidia-uvm','nvidia-uvm-tools']:
        p=Path('/dev')/name
        if p.exists():cmd+=['--dev-bind',str(p),str(p)]
    for name in ['driver/nvidia','cpuinfo','meminfo']:
        p=Path('/proc')/name
        if p.exists():cmd+=['--ro-bind',str(p),str(p)]
    cmd+=['--clearenv','--setenv','PATH',str(ROOT/'env/bin')+':/usr/bin',
        '--setenv','LD_PRELOAD',str(pybase/'lib/libpython3.8.so.1.0'),'--setenv','LD_LIBRARY_PATH',str(pybase/'lib')+':'+str(ROOT/'env/lib/python3.8/site-packages/torch/lib'),'--setenv','HOME','/tmp','--setenv','PYTHONDONTWRITEBYTECODE','1',
        '--setenv','OMP_NUM_THREADS','2','--setenv','MKL_NUM_THREADS','2',
        '--setenv','OPENBLAS_NUM_THREADS','2','--setenv','M8A_ISOLATED','1',
        '--setenv','PETR_SOURCE',str(ROOT/'sources/PETR'),
        '--setenv','PYTHONPATH',str(ROOT/'sources/mmdetection3d'),
        '--setenv','CUDA_VISIBLE_DEVICES','0','--chdir','/tmp',str(ROOT/'env/bin/python'),'-s','/gpu_probe.py']
    return cmd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--extract-only',action='store_true');args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    if args.extract_only:
        run('extract_'+args.run_id,command(out,Path('tools/m8a_p1_safe_extract.py')),timeout=120)
        return
    import json
    extraction=json.loads((ROOT/'artifacts/safe_extract/safe_extract.json').read_text())
    cmd=['unshare','--net',str(ROOT/'env/bin/python'),'-s',str(Path('tools/m8a_p1_gpu_probe.py').resolve())]
    runtime_env=dict(M8A_ISOLATED='1',M8A_OUTPUT=str(out),PETR_SOURCE=str(ROOT/'sources/PETR'),
        PYTHONPATH=str(ROOT/'sources/mmdetection3d'),M8A_SAFE_WEIGHTS=str(ROOT/'artifacts/safe_extract/model_tensors.npz'),
        M8A_SAFE_SHA256=extraction['sha256'],OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(ROOT/'env/lib/python3.8/site-packages/torch/lib')+':'+str(ROOT/'cuda-11.8/lib'))
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            samples.append(dict(monotonic_s=time.monotonic(),result=p.stdout.strip(),returncode=p.returncode))
            stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:run('gpu_'+args.run_id,cmd,timeout=600,extra_env=runtime_env)
    finally:
        stop.set();thread.join()
        write(LOG/('gpu_monitor_'+args.run_id+'.json'),dict(**provenance('ActualExternalGPUProcessSampling'),samples=samples,interval_s=.25))
        write(LOG/('gpu_artifacts_'+args.run_id+'.json'),dict(**provenance('IsolatedGPUArtifactManifest'),files=[asset(p) for p in sorted(out.iterdir())]))

if __name__=='__main__':main()
