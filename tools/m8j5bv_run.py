"""Append-only bounded validation runner; never updates real model weights."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT=Path(os.environ.get('M8J5BV_PRIVATE_ROOT','/root/gpufree-data/hs-drive-private/m8j5bv-full-validation'))
PUBLIC=Path(os.environ.get('M8J5BV_REPORT_ROOT','reports'))
REPORT=PUBLIC/'m8j5bv_sources'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['baseline','synthetic','replay']);ap.add_argument('--suffix',default='run1');args=ap.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
    runtime=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    env=dict(PATH=str(runtime/'env/bin')+':/usr/bin:/bin',HOME=str(ROOT),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(runtime/'env/lib/python3.8/site-packages/torch/lib')+':'+str(runtime/'cuda-11.8/lib'),
        PYTHONPATH=str(Path.cwd())+':'+str(runtime/'sources/PETR')+':'+str(runtime/'sources/mmdetection3d'),
        M8J5_RUNTIME=str(runtime),M8J5_PRIVATE='/root/gpufree-data/hs-drive-private/m8j5-supervision',
        M8J5B_PRIVATE='/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision',
        M8J5B_TEST_REPORT=str(REPORT/('baseline_gradients_'+args.suffix+'.json')),
        M8J5BV_REPORT=str(REPORT/('extended_gradients_'+args.suffix+'.json')),
        M8J5BV_PRIVATE_ROOT=str(ROOT.resolve()),M8J5BV_REPORT_ROOT=str(PUBLIC.resolve()),
        M8J5BV_SUFFIX=args.suffix,MPLCONFIGDIR=str(ROOT/'mplconfig'))
    command=['unshare','--net',str(runtime/'env/bin/python'),'-m']
    if args.mode=='baseline':command+=['pytest','--noconftest','tests/test_m8j5b_petr_loss.py','tests/test_m8j5_petr_loss.py','-q','-ra']
    elif args.mode=='synthetic':command+=['tools.m8j5bv_synthetic_entry']
    else:command+=['tools.m8j5bv_replay']
    sources=[p for pattern in ('training/m8j5bv*','tests/test_m8j5bv*','tools/m8j5bv*') for p in Path('.').glob(pattern) if p.is_file()]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started=datetime.now(timezone.utc).isoformat()
    start=time.perf_counter();log=REPORT/(args.mode+'_'+args.suffix+'.log')
    with log.open('x') as f:
        proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
    result=dict(command=command,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=rc,elapsed_s=time.perf_counter()-start,network='Disabled',real_optimizer_steps=0,
        source_sha256_at_launch=source_hashes,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    with (REPORT/(args.mode+'_'+args.suffix+'_execution.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));raise SystemExit(rc)


if __name__=='__main__':main()
