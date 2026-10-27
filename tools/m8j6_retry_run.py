"""Single-run supervisor: two updates, new-process restore+one, new verify."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from datasets.m8j6_selection import sha256


def main():
    repo=Path.cwd();run=json.loads(Path('configs/m8j6_retry_protocol.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Commit code and tests before the real run')
    rt=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    private=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini')/run['run_id']
    reports=repo/'reports/m8j6_retry_sources'/run['run_id']
    private.mkdir(mode=0o700,exist_ok=False);reports.mkdir(exist_ok=False)
    base_env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),
        PYTHONPATH=os.pathsep.join(map(str,[repo,rt/'sources/PETR',rt/'sources/mmdetection3d'])),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',M8J6_RUNTIME=str(rt),M8J6R_ROOT=str(private),M8J6R_COMMIT=commit)
    record=dict(run_id=run['run_id'],code_commit=commit,base_commit=run['base_commit'],
        started_at_utc=datetime.now(timezone.utc).isoformat(),old_run_optimizer_steps=1,phases=[],
        source_sha256={str(p):sha256(p) for pattern in ('tools/m8j6_retry*','training/m8j6_retry*','configs/m8j6_retry*','tests/test_m8j6_retry*') for p in repo.glob(pattern)},
        no_network=True,supervisor_pid=os.getpid(),automatic_retry=False)
    exitcode=0
    for phase in ('first','resume','verify'):
        rep=reports/phase;rep.mkdir(exist_ok=False)
        env=dict(base_env,M8J6R_REPORT=str(rep),M8J6R_PHASE=phase)
        command=['unshare','--net',str(rt/'env/bin/python'),'-m','tools.m8j6_retry_gpu']
        start=time.monotonic()
        with (rep/'execution.txt').open('x') as log:
            process=subprocess.Popen(command,env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();rc=124
        phase_record=dict(phase=phase,command=command,process_pid=process.pid,exit_code=rc,
            elapsed_s=time.monotonic()-start,log_sha256=sha256(rep/'execution.txt'),code_commit=commit)
        (rep/'execution.json').write_text(json.dumps(phase_record,indent=2))
        record['phases'].append(phase_record)
        print(json.dumps(phase_record),flush=True)
        if rc:
            exitcode=rc;break
    results=[json.loads((reports/p['phase']/'result.json').read_text()) for p in record['phases'] if (reports/p['phase']/'result.json').exists()]
    ledgers=list(reports.glob('*/optimizer_step_*_returned.json'))
    record['new_run_optimizer_steps']=len(ledgers)
    record['all_attempts_optimizer_steps']=1+len(ledgers)
    record['status']='Passed' if (exitcode==0 and len(results)==3 and len(ledgers)==3 and
        [r['optimizer_calls_this_process'] for r in results]==[2,1,0] and
        all(r['status']=='PhasePassed' for r in results) and results[-1]['restore']['passed']) else 'FailedStopped'
    record['private_checkpoints_published']=False
    record['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    (reports/'supervisor.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
    return 0 if record['status']=='Passed' else (exitcode or 1)


if __name__=='__main__':raise SystemExit(main())
