"""Bounded isolated launcher, reusing the installed J5BV official PETR environment."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='run1')
    ap.add_argument('--legacy-determinism',action='store_true',
        help='Explicit diagnostic fallback only after retained torch2.0 deterministic indexing failure')
    ap.add_argument('--minimal-dcn',action='store_true',help='Standalone synthetic DCNv2, no whole-model construction')
    args=ap.parse_args()
    rt=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    root=Path('/root/gpufree-data/hs-drive-private/m8j5c-numerical-supervision')/args.name
    root.mkdir(parents=True,exist_ok=False)
    out=Path('reports/m8j5c_sources')/args.name;out.mkdir(parents=True,exist_ok=False)
    env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(root),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),
        PYTHONPATH=str(Path.cwd())+':'+str(rt/'sources/PETR')+':'+str(rt/'sources/mmdetection3d'),
        M8J5_RUNTIME=str(rt),M8J5C_ROOT=str(root),M8J5C_OUTPUT=str(out.resolve()),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['M8J5C_DETERMINISTIC']='0' if args.legacy_determinism else '1'
    env['M8J5C_DETERMINISM_REASON']='RetainedRun1OfficialTorch2DeterministicScalarIndexAssignmentFailure' if args.legacy_determinism else 'None'
    command=['unshare','--net',str(rt/'env/bin/python'),'-m',
             'tools.m8j5c_dcn_minimal' if args.minimal_dcn else 'tools.m8j5c_numeric']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('tools/m8j5c*','configs/m8j5c*')
            for p in Path('.').glob(pattern) if p.is_file()}
    start=time.perf_counter();stamp=datetime.now(timezone.utc).isoformat()
    with (out/'execution.log').open('x') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=600)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    record=dict(command=command,started_at_utc=stamp,exit_code=rc,elapsed_s=time.perf_counter()-start,
        source_sha256=hashes,log_sha256=hashlib.sha256((out/'execution.log').read_bytes()).hexdigest(),
        real_data_used=False,optimizer_steps=0,network='Disabled',private_root=str(root),
        torch_deterministic_algorithms=not args.legacy_determinism,override_reason=env['M8J5C_DETERMINISM_REASON'])
    (out/'execution.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record));raise SystemExit(rc)


if __name__=='__main__':main()
