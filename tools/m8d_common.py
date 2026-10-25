"""M8-D append-only evidence and isolated execution helpers."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tools.m8a_common import asset, sha, write
from tools.m8b_common import environment

CFG = json.loads(Path('configs/m8d/protocol.json').read_text())
ROOT = Path(CFG['root'])
BASELINE = Path(CFG['baseline_root'])
M8C = Path(CFG['m8c_root'])
RUNTIME = Path(CFG['runtime'])
LOG = Path('reports/m8d_logs')


def provenance(kind):
    paths = [Path('configs/m8d/protocol.json')]
    for directory, pattern in [('tools', 'm8d_*.py'), ('datasets', 'm8d_*.py'),
                               ('diagnostics', 'm8d_*.py'), ('tests', 'test_m8d*.py')]:
        paths.extend(sorted(Path(directory).glob(pattern)))
    return dict(kind=kind, base=CFG['base'],
                execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                sources={str(p): sha(p) for p in paths},
                checkpoint_sha256=CFG['checkpoint_sha256'], petr_commit=CFG['petr_commit'],
                training_steps=0, visual_control=False)


def run(name, command, timeout=1800):
    env = environment()
    env['HOME'] = str(ROOT / 'home')
    start = time.perf_counter()
    with (LOG / (name + '.txt')).open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    write(LOG / (name + '.json'), dict(**provenance('ActualServerProcess'), command=command,
          returncode=code, elapsed_s=time.perf_counter()-start, log=asset(LOG / (name+'.txt'))))
    if code:
        raise RuntimeError('Failure retained: ' + name)
    print(name, 'completed', flush=True)


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float).ravel()
    if not len(a):
        return dict(n=0, mean=None, p50=None, p90=None, p95=None, p99=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p90=float(np.percentile(a,90)), p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)), max=float(a.max()))
