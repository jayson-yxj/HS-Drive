"""Bounded, network-isolated preparation and A/B evaluation; no training."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import signal
import time
from datasets.m8j6_selection import sha256


def main():
    cfg=json.loads(Path('configs/m8j8_protocol.json').read_text());repo=Path.cwd()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit frozen files first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    rt=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    root=Path(cfg['prepared_root']).parent;root.mkdir(mode=0o700,exist_ok=False)
    report=repo/'reports/m8j8_sources'/cfg['run_id'];report.mkdir(exist_ok=False)
    env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(root),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),
        PYTHONPATH=os.pathsep.join(map(str,[repo,rt/'sources/PETR',rt/'sources/mmdetection3d'])),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',M8J6_RUNTIME=str(rt),M8J8_COMMIT=commit)
    result=dict(run_id=cfg['run_id'],code_commit=commit,started_at_utc=datetime.now(timezone.utc).isoformat(),phases=[],
        optimizer_steps=0,no_network=True,automatic_retry=False,
        source_sha256={str(p):sha256(p) for pattern in ['tools/m8j8*.py','training/m8j8*.py','configs/m8j8*','tests/test_m8j8*'] for p in repo.glob(pattern)})
    rc=0
    for phase in ('prepare','A','B'):
        rep=report/phase;rep.mkdir(exist_ok=False)
        module='tools.m8j8_prepare' if phase=='prepare' else 'tools.m8j8_gpu'
        e=dict(env,M8J8_PHASE=phase,M8J8_REPORT=str(rep))
        if phase=='prepare':e['CUDA_VISIBLE_DEVICES']=''
        command=['unshare','--net',str(rt/'env/bin/python'),'-m',module];start=time.monotonic()
        with (rep/'execution.txt').open('x') as f:
            child=subprocess.Popen(command,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=child.wait(timeout=1200)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();rc=124
        record=dict(phase=phase,PID=child.pid,exit_code=rc,elapsed_s=time.monotonic()-start,command=command,
            log_sha256=sha256(rep/'execution.txt'))
        (rep/'execution.json').write_text(json.dumps(record,indent=2));result['phases'].append(record)
        print(json.dumps(record),flush=True)
        if rc:break
    result.update(status='ReadOnlyABCompleted' if rc==0 and len(result['phases'])==3 else 'FailedStopped',
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    (report/'supervisor.json').write_text(json.dumps(result,indent=2))
    return rc


if __name__=='__main__':raise SystemExit(main())
