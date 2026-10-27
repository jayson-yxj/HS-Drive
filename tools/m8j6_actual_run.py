"""Bounded network-isolated execution in the unchanged PETR environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['prepare', 'baseline', 'smoke'])
    ap.add_argument('--name', help='Distinct retained attempt identifier')
    args = ap.parse_args()
    repo = Path.cwd(); rt = Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    root = Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini')
    report = repo / 'reports/m8j6_sources' / ('actual_' + (args.name or args.phase))
    report.mkdir(exist_ok=False)
    env = dict(PATH=str(rt / 'env/bin') + ':/usr/bin:/bin', HOME=str(root), LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        LD_LIBRARY_PATH=str(rt / 'env/lib/python3.8/site-packages/torch/lib') + ':' + str(rt / 'cuda-11.8/lib'),
        PYTHONPATH=os.pathsep.join(map(str, [repo, rt / 'sources/PETR', rt / 'sources/mmdetection3d'])),
        M8J6_ROOT=str(root), M8J6_RUNTIME=str(rt), M8J6_REPORT=str(report), M8J6_PHASE=args.phase)
    if args.phase == 'prepare':
        env['CUDA_VISIBLE_DEVICES'] = ''
    module = 'tools.m8j6_prepare_actual' if args.phase == 'prepare' else 'tools.m8j6_gpu_actual'
    command = ['unshare', '--net', str(rt / 'env/bin/python'), '-m', module]
    start = time.monotonic(); stamp = datetime.now(timezone.utc).isoformat()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for pattern in ('tools/m8j6*', 'datasets/m8j6*', 'configs/m8j6*', 'training/m8j6*', 'tests/test_m8j6*')
              for p in repo.glob(pattern) if p.is_file()}
    with (report / 'execution.txt').open('x') as log:
        p = subprocess.Popen(command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 124
    record = dict(phase=args.phase, command=command, started_at_utc=stamp, exit_code=rc,
                  elapsed_s=time.monotonic() - start, source_sha256=hashes,
                  log_sha256=hashlib.sha256((report / 'execution.txt').read_bytes()).hexdigest(),
                  network='Disabled', timeout_s=900, default_environment_modified=False)
    (report / 'execution.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2)); return rc


if __name__ == '__main__':
    raise SystemExit(main())
