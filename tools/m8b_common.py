"""M8-B append-only evidence and independent experiment scope."""
import json,os,signal,subprocess,time
from pathlib import Path
from tools.m8a_common import asset,sha,write
CFG=json.loads(Path('configs/m8b/protocol.json').read_text())
ROOT=Path(CFG['root']);RUNTIME=Path(CFG['runtime']);LOG=Path('reports/m8b_logs')
def provenance(kind):
    paths=[Path('configs/m8b/protocol.json'),*sorted(Path('tools').glob('m8b_*.py')),*sorted(Path('datasets').glob('m8b_*.py')),*sorted(Path('diagnostics').glob('m8b_*.py')),*sorted(Path('tests').glob('test_m8b*.py'))]
    return dict(kind=kind,base=CFG['base'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in paths},training_steps=0,visual_control=False)
def environment():
    return dict(PATH=str(RUNTIME/'env/bin')+':/usr/bin:/bin',HOME=str(ROOT/'home'),LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
      LD_LIBRARY_PATH=str(RUNTIME/'env/lib/python3.8/site-packages/torch/lib')+':'+str(RUNTIME/'cuda-11.8/lib'),PYTHONPATH=str(Path.cwd())+':'+str(RUNTIME/'sources/PETR')+':'+str(RUNTIME/'sources/mmdetection3d'))
def run(name,cmd,timeout=600,env=None):
    start=time.perf_counter();LOG.mkdir(exist_ok=True)
    with (LOG/(name+'.txt')).open('x') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env or environment(),start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    write(LOG/(name+'.json'),dict(**provenance('ActualServerSubprocess'),command=cmd,exit_code=rc,elapsed_s=time.perf_counter()-start,log=asset(LOG/(name+'.txt'))))
    if rc:raise RuntimeError(name+' failed; log retained')
    print(name,'Passed',flush=True)
