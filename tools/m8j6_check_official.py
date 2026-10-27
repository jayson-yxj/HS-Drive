"""Offline CPU contract check in existing PETR environment; no model/GPU run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    rt = Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [repo, rt / 'sources/PETR', rt / 'sources/mmdetection3d'])),
               M8J6_RUNTIME=str(rt), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    command = ['unshare', '--net', str(rt / 'env/bin/python'), '-m', 'pytest', '--noconftest',
               '-q', 'tests/test_m8j6_official_contract.py']
    stamp = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with (args.output / 'tests.txt').open('x') as f:
        try:
            result = subprocess.run(command, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=90)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    record = dict(command=command, started_at_utc=stamp, exit_code=rc,
                  elapsed_s=time.monotonic() - start, scope='SyntheticCPU_OfficialFrozenImplementations',
                  network='Disabled', CUDA_VISIBLE_DEVICES='', model_loaded=False, optimizer_steps=0,
                  conftest='Disabled: main project conftest uses Python3.10 syntax; frozen PETR uses Python3.8',
                  log_sha256=hashlib.sha256((args.output / 'tests.txt').read_bytes()).hexdigest())
    (args.output / 'execution.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
