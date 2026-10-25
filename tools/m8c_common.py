"""Append-only, versioned M8-C diagnostics; no default model changes."""
import json,os,subprocess,signal,time
from pathlib import Path
from tools.m8a_common import asset,sha,write
CFG=json.loads(Path('configs/m8c/protocol.json').read_text());ROOT=Path(CFG['root']);BASELINE=Path(CFG['baseline_root']);RUNTIME=Path(CFG['runtime']);LOG=Path('reports/m8c_logs')
def provenance(kind):
    paths=[Path('configs/m8c/protocol.json'),*sorted(Path('tools').glob('m8c_*.py')),*sorted(Path('diagnostics').glob('m8c_*.py')),*sorted(Path('tests').glob('test_m8c*.py'))]
    return dict(kind=kind,base=CFG['base'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in paths},checkpoint_sha256=CFG['checkpoint_sha256'],training_steps=0,visual_control=False)
def env():
    from tools.m8b_common import environment
    e=environment();e['HOME']=str(ROOT/'home');return e
def run(name,cmd,timeout=600):
    t=time.perf_counter()
    with (LOG/(name+'.txt')).open('x') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env(),start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    write(LOG/(name+'.json'),dict(**provenance('ActualServerProcess'),command=cmd,returncode=rc,elapsed_s=time.perf_counter()-t,log=asset(LOG/(name+'.txt'))))
    print(name,rc,flush=True)
    if rc:raise RuntimeError('Retained failure: '+name)
def stats(x):
    import numpy as np
    a=np.asarray(x,float)
    return dict(n=int(a.size),min=float(a.min()) if a.size else None,p50=float(np.median(a)) if a.size else None,p90=float(np.percentile(a,90)) if a.size else None,p95=float(np.percentile(a,95)) if a.size else None,max=float(a.max()) if a.size else None,mean=float(a.mean()) if a.size else None)
