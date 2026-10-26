"""Bounded isolated runtime runner, no installs/downloads or real optimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tests','forward']);ap.add_argument('--suffix',default='run1')
    args=ap.parse_args()
    runtime=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    private=Path(json.loads(Path('configs/m8j5_protocol.json').read_text())['private_root'])
    evidence=Path('reports/m8j5_sources');evidence.mkdir(exist_ok=True)
    env=dict(PATH=str(runtime/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(runtime/'env/lib/python3.8/site-packages/torch/lib')+':'+str(runtime/'cuda-11.8/lib'),
        PYTHONPATH=str(Path.cwd())+':'+str(runtime/'sources/PETR')+':'+str(runtime/'sources/mmdetection3d'),
        M8J5_RUNTIME=str(runtime),M8J5_PRIVATE=str(private),
        M8J5_FORWARD_SUBDIR='forward_'+args.suffix,
        MPLCONFIGDIR=str(private/('forward_'+args.suffix)/'mplconfig'),
        M8J5_TEST_REPORT=str(evidence/('gradient_'+args.suffix+'.json')))
    command=['unshare','--net',str(runtime/'env/bin/python'),'-m']
    command+=['pytest','--noconftest','tests/test_m8j5_petr_loss.py','-q','-ra'] if args.mode=='tests' else ['tools.m8j5_forward_regression']
    log=evidence/(args.mode+'_'+args.suffix+'.log');start=time.perf_counter()
    with log.open('x') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=240)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    result=dict(command=command,exit_code=rc,elapsed_s=time.perf_counter()-start,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),network='Disabled',real_optimizer_steps=0,
        source_sha256=hashlib.sha256(Path('training/m8j5_ignore_loss.py').read_bytes()).hexdigest())
    with (evidence/(args.mode+'_'+args.suffix+'_execution.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));raise SystemExit(rc)


if __name__=='__main__':main()
