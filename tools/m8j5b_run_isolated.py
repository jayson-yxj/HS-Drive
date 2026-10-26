"""Bounded existing PETR environment execution, network off, no installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tests','diagnostic']);ap.add_argument('--suffix',default='run1');args=ap.parse_args()
    runtime=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    private=Path(os.environ.get('HS_M8J5B_ROOT','/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision'))
    evidence=Path('reports/m8j5b_sources')
    env=dict(PATH=str(runtime/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(runtime/'env/lib/python3.8/site-packages/torch/lib')+':'+str(runtime/'cuda-11.8/lib'),
        PYTHONPATH=str(Path.cwd())+':'+str(runtime/'sources/PETR')+':'+str(runtime/'sources/mmdetection3d'),
        M8J5_RUNTIME=str(runtime),M8J5_PRIVATE='/root/gpufree-data/hs-drive-private/m8j5-supervision',M8J5B_PRIVATE=str(private),
        M8J5B_TEST_REPORT=str(evidence/('gradients_'+args.suffix+'.json')),
        M8J5B_DIAGNOSTIC_OUTPUT=str(private/('predictions_'+args.suffix)),MPLCONFIGDIR=str(private/'mplconfig'))
    command=['unshare','--net',str(runtime/'env/bin/python'),'-m']
    command+=['pytest','--noconftest','tests/test_m8j5b_petr_loss.py','tests/test_m8j5_petr_loss.py','-q','-ra'] if args.mode=='tests' else ['tools.m8j5b_query_diagnostic']
    start=time.perf_counter();log=evidence/(args.mode+'_'+args.suffix+'.log')
    with log.open('x') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=300)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    sources=[p for pattern in ('training/m8j5b*.py','datasets/m8j5b*.py','tools/m8j5b*.py','tests/test_m8j5b*.py') for p in Path('.').glob(pattern)]
    result=dict(command=command,exit_code=rc,elapsed_s=time.perf_counter()-start,network='Disabled',optimizer_steps=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    with (evidence/(args.mode+'_'+args.suffix+'_execution.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));raise SystemExit(rc)


if __name__=='__main__':main()
