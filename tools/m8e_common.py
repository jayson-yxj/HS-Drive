"""Append-only M8-E data/evaluator experiment helpers."""
import json, os, signal, subprocess, time
from pathlib import Path
from tools.m8a_common import asset,sha,write
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8e-groundtruth')
BASE='037ddba29d6eaaa292e8ad74db58bd94e5f42b50'
LOG=Path('reports/m8e_logs')
def provenance(kind):
    paths=[*Path('configs/m8e').glob('*.json'),*Path('tools').glob('m8e_*.py'),
           *Path('datasets').glob('m8e_*.py'),*Path('diagnostics').glob('m8e_*.py'),
           *Path('tests').glob('test_m8e*.py')]
    return dict(kind=kind,base=BASE,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={str(p):sha(p) for p in sorted(paths)},training_steps=0,visual_control=False)
def run(name,command,timeout=1800,environment=None):
    start=time.perf_counter()
    with (LOG/(name+'.txt')).open('x') as out:
        env=dict(os.environ if environment is None else environment,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
    write(LOG/(name+'.json'),dict(**provenance('ActualServerProcess'),command=command,returncode=code,
          elapsed_s=time.perf_counter()-start,log=asset(LOG/(name+'.txt'))))
    if code:raise RuntimeError('Failure retained: '+name)
    print(name,'completed',flush=True)
