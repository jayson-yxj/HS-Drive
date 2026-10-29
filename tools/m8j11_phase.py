"""Record bounded phase invocation/exit status without overwriting logs."""
import argparse,hashlib,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--timeout',type=int,default=600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.command and a.timeout<=1800
out=Path('reports/m8j11_sources')/a.name
record=dict(started_utc=datetime.now(timezone.utc).isoformat(),command=a.command,timeout_s=a.timeout,
 execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('tools/m8j11*','datasets/m8j11*','configs/m8j11*') for p in Path('.').glob(pattern) if p.is_file()})
t=time.monotonic()
with out.with_suffix('.txt').open('x') as f:
 q=subprocess.Popen(a.command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 record['PID']=q.pid
 try:rc=q.wait(timeout=a.timeout)
 except subprocess.TimeoutExpired:os.killpg(q.pid,signal.SIGKILL);q.wait();rc=124
record.update(exit_code=rc,elapsed_s=time.monotonic()-t,log_sha256=hashlib.sha256(out.with_suffix('.txt').read_bytes()).hexdigest())
with out.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2));raise SystemExit(rc)
