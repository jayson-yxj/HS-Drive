"""Launch one bounded training run and a fresh-process read-only final reload."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datasets.m8j6_selection import sha256


def main():
    repo=Path.cwd();cfg=json.loads(Path('configs/m8j7_protocol.json').read_text())
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit frozen code and tests first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    rt=Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    private=Path('/root/gpufree-data/hs-drive-private')/cfg['run_id']
    report=repo/'reports/m8j7_sources'/cfg['run_id']
    private.mkdir(mode=0o700,exist_ok=False);report.mkdir(exist_ok=False)
    env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),
        PYTHONPATH=os.pathsep.join(map(str,[repo,rt/'sources/PETR',rt/'sources/mmdetection3d'])),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',M8J6_RUNTIME=str(rt),M8J7_PRIVATE=str(private),M8J7_COMMIT=commit)
    record=dict(run_id=cfg['run_id'],code_commit=commit,started_at_utc=datetime.now(timezone.utc).isoformat(),
        no_network=True,automatic_retry=False,private_checkpoints_published=False,phases=[],
        source_sha256={str(p):sha256(p) for pattern in ['tools/m8j7*','training/m8j7*','configs/m8j7*','tests/test_m8j7*'] for p in repo.glob(pattern)})
    rc=0
    for phase in ('train','verify'):
        out=report/phase;out.mkdir(exist_ok=False)
        child_env=dict(env,M8J7_REPORT=str(out),M8J7_PHASE=phase,M8J6R_PHASE='m8j7_'+phase)
        command=['unshare','--net',str(rt/'env/bin/python'),'-m','tools.m8j7_gpu']
        start=time.monotonic()
        with (out/'execution.txt').open('x') as log:
            child=subprocess.Popen(command,cwd=repo,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=child.wait(timeout=cfg['limits']['timeout_seconds'] if phase=='train' else 120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();rc=124
        phase_record=dict(phase=phase,PID=child.pid,exit_code=rc,elapsed_s=time.monotonic()-start,
            command=command,log_sha256=sha256(out/'execution.txt'))
        (out/'execution.json').write_text(json.dumps(phase_record,indent=2))
        record['phases'].append(phase_record);print(json.dumps(phase_record),flush=True)
        if rc:break
    ledgers=list((private/'steps').glob('optimizer_step_*_returned.json'))
    record['actual_optimizer_steps']=len(ledgers)
    record['status']='CompletedAndReloadPassed' if rc==0 and len(ledgers)==400 and len(record['phases'])==2 else 'FailedStopped'
    record['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    (report/'supervisor.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
    return 0 if record['status']=='CompletedAndReloadPassed' else (rc or 1)


if __name__=='__main__':raise SystemExit(main())
