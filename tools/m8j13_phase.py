"""Bounded command evidence without overwriting earlier attempts."""
import argparse,os,signal,subprocess,time
from pathlib import Path
from tools.m8j13_common import config,write,digest,provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('--timeout',type=float,default=600);p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    os.umask(0o077);root=Path(config()['private_root']);log=root/(a.name+'.log');record=dict(command=a.command,provenance=provenance(),timeout_s=a.timeout)
    started=time.monotonic()
    with log.open('x') as f:
        proc=subprocess.Popen(a.command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rc=124
    record.update(exit_code=rc,elapsed_s=time.monotonic()-started,log=str(log),log_sha256=digest(log))
    write(Path('reports/m8j13_sources')/(a.name+'_execution.json'),record)
    print(record,flush=True);raise SystemExit(rc)


if __name__=='__main__':main()
