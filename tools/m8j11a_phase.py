"""Bounded append-only command evidence; child logs stay private."""
import argparse
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from tools.m8j11a_common import read,write,sha,PROTOCOL

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cfg=read(PROTOCOL);root=Path(cfg['private_root']);log=root/(a.name+'.txt')
    start=time.monotonic()
    record=dict(command=a.command,host_utc=datetime.now(timezone.utc).isoformat(),client_task_date=cfg['client_date'],
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256={str(p):sha(p) for pat in ('tools/m8j11a*','datasets/m8j11a*','configs/m8j11a*','tests/m8j11a/*.py') for p in Path('.').glob(pat) if p.is_file()},timeout_s=600)
    with log.open('x') as f:
        q=subprocess.Popen(a.command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=q.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(q.pid,signal.SIGKILL);q.wait();rc=124
    record.update(exit_code=rc,elapsed_s=time.monotonic()-start,log_path=str(log),log_sha256=sha(log))
    write('reports/m8j11a_sources/'+a.name+'.json',record)
    print(a.name,'exit',rc,'elapsed',record['elapsed_s'],flush=True)
    raise SystemExit(rc)

if __name__=='__main__':main()
