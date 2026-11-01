"""Bounded isolated task command with source identities and private full logs."""
import argparse,json,subprocess,os,time,signal,hashlib
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--timeout',type=int,default=600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cfg=json.loads(Path('configs/m8j12_protocol.json').read_text());root=Path(cfg['private_root']);log=root/(a.name+'.log')
    assert not log.exists();os.umask(0o077)
    record={'command':a.command,'utc':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'timeout_s':a.timeout,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for pat in ['tools/m8j12*','datasets/m8j12*','configs/m8j12*','tests/m8j12/*.py'] for f in Path('.').glob(pat) if f.is_file()}}
    start=time.monotonic()
    with log.open('x') as f:
        proc=subprocess.Popen(a.command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rc=124
    record.update(exit_code=rc,elapsed_s=time.monotonic()-start,log_path=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    out=Path('reports/m8j12_sources')/(a.name+'_execution.json')
    with out.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record));raise SystemExit(rc)
if __name__=='__main__':main()
