"""Sequential bounded CPU export and two independent official evaluator runs."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    repo=Path.cwd();cfg=json.loads(Path('configs/m8j9_protocol.json').read_text());rt=Path(cfg['runtime'])
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit frozen scope/code first'
    private=Path(cfg['private_output']);private.mkdir(mode=0o700,exist_ok=False)
    report=repo/'reports/m8j9_sources'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),
        PYTHONPATH=os.pathsep.join(map(str,[repo,rt/'sources/PETR',rt/'sources/mmdetection3d'])),
        M8J9_COMMIT=commit)
    summary=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),execution_commit=commit,phases=[],
        model_forward_calls=0,optimizer_steps=0,network=False,GPU=False,automatic_retry=False,
        source_sha256={str(p):sha(p) for pattern in ('tools/m8j9*.py','datasets/m8j9*.py','configs/m8j9*','tests/test_m8j9*') for p in repo.glob(pattern)})
    rc=0
    for phase in ('export','eval1','eval2'):
        out=report/phase;out.mkdir(exist_ok=False)
        command=['unshare','--net',str(rt/'env/bin/python'),'-m','tools.m8j9_stage']
        e=dict(env,M8J9_PHASE=phase,M8J9_REPORT=str(out));start=time.monotonic()
        with (out/'execution.txt').open('x') as f:
            proc=subprocess.Popen(command,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
        entry=dict(phase=phase,PID=proc.pid,exit_code=rc,elapsed_s=time.monotonic()-start,
                   command=command,timeout_s=600,log_sha256=sha(out/'execution.txt'))
        summary['phases'].append(entry);print(json.dumps(entry),flush=True)
        if rc:break
    summary.update(status='ExportAndTwoOfficialEvaluationsCompleted' if len(summary['phases'])==3 and rc==0 else 'FailedStopped',
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    with (report/'supervisor.json').open('x') as f:json.dump(summary,f,indent=2)
    return rc


if __name__=='__main__':raise SystemExit(main())
