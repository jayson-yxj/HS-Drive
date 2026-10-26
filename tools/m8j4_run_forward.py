"""Run three frozen single-frame forwards in existing network-isolated runtime."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    cfg=json.loads(Path('configs/m8j4_protocol.json').read_text())
    root=Path(cfg['private_root']);runtime=Path(cfg['model']['runtime'])
    out=root/'forward';out.mkdir(exist_ok=False)
    log=Path('reports/m8j4_sources/gpu_forward.log')
    env=dict(PATH=str(runtime/'env/bin')+':/usr/bin:/bin',HOME=str(out),
        LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(runtime/'env/lib/python3.8/site-packages/torch/lib')+':'+str(runtime/'cuda-11.8/lib'),
        PYTHONPATH=str(Path.cwd())+':'+str(runtime/'sources/PETR')+':'+str(runtime/'sources/mmdetection3d'),
        M8J4_INPUT=str(root/'inference_input'),M8J4_OUTPUT=str(out),M8J4_RUNTIME=str(runtime))
    command=['unshare','--net',str(runtime/'env/bin/python'),'-m','tools.m8j4_forward']
    start=time.perf_counter()
    with log.open('x') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
    record=dict(command=command,exit_code=rc,elapsed_s=time.perf_counter()-start,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forward_source_sha256=hashlib.sha256(Path('tools/m8j4_forward.py').read_bytes()).hexdigest(),
        new_dependencies=False,default_model_changes=False,training_steps=0,
        isolation='Network namespace + sanitized numeric RGB bundle + Python open canary. Not hostile-native-code filesystem sandbox.')
    with Path('reports/m8j4_sources/gpu_execution.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record))
    if rc:raise SystemExit(rc)


if __name__=='__main__':main()
