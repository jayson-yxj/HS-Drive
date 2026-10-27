"""Explicit single-phase runner; bounded, isolated, no automatic restart."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from datasets.m8j6_selection import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','verify','official','report']);args=parser.parse_args()
    cfg=json.loads(Path('configs/m8j10_protocol.json').read_text());repo=Path.cwd();rt=Path(cfg['runtime'])
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit frozen code first'
    private=Path(cfg['private_root']);private.mkdir(mode=0o700,exist_ok=True)
    rep=repo/'reports/m8j10_sources'/cfg['run_id']/args.phase;rep.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    module={'prepare':'tools.m8j10_prepare','train':'tools.m8j10_gpu','verify':'tools.m8j10_gpu','official':'tools.m8j10_official','report':'tools.m8j10_report'}[args.phase]
    timeout=cfg['limits']['timeout_train_s'] if args.phase=='train' else cfg['limits']['timeout_prepare_s'] if args.phase=='prepare' else cfg['limits']['timeout_evaluation_s']
    env=dict(PATH=str(rt/'env/bin')+':/usr/bin:/bin',HOME=str(private),LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LD_LIBRARY_PATH=str(rt/'env/lib/python3.8/site-packages/torch/lib')+':'+str(rt/'cuda-11.8/lib'),PYTHONPATH=os.pathsep.join(map(str,[repo,rt/'sources/PETR',rt/'sources/mmdetection3d'])),CUBLAS_WORKSPACE_CONFIG=':4096:8',M8J6_RUNTIME=str(rt),M8J10_REPORT=str(rep),M8J10_PHASE=args.phase,M8J10_COMMIT=commit,M8J6R_PHASE='m8j10_'+args.phase)
    if args.phase not in ('train','verify'):env['CUDA_VISIBLE_DEVICES']=''
    record=dict(phase=args.phase,code_commit=commit,started_at_utc=datetime.now(timezone.utc).isoformat(),timeout_s=timeout,network=False,automatic_retry=False,source_sha256={str(p):sha256(p) for pat in ('tools/m8j10*.py','datasets/m8j10*.py','training/m8j10*.py','configs/m8j10*.json','reports/m8j10_training_manifest.json','tests/test_m8j10*.py') for p in repo.glob(pat)})
    command=['unshare','--net',str(rt/'env/bin/python'),'-m',module];start=time.monotonic()
    with (rep/'execution.txt').open('x') as f:
        proc=subprocess.Popen(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
    record.update(PID=proc.pid,exit_code=rc,elapsed_s=time.monotonic()-start,command=command,log_sha256=sha256(rep/'execution.txt'),finished_at_utc=datetime.now(timezone.utc).isoformat(),status='Completed' if rc==0 else 'FailedStopped',actual_optimizer_steps=len(list((private/'steps').glob('optimizer_step_*_returned.json'))))
    (rep/'supervisor.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
    return rc


if __name__=='__main__':raise SystemExit(main())
