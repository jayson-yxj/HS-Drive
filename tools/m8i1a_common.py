"""Independent M8-I1a durable artifacts; never writes M8-I1 assets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
BASE = '916ba75072259f17da725ddbc9c76f2f5fbeea13'
PROTOCOL = REPO / 'configs/m8i1a/protocol.json'
OLD_ROOT = Path('/root/gpufree-data/hs-drive-experiments/m8i1-traffic-development')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def asset(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def persist(path, value):
    """Exclusive creation, data fsync, then directory fsync before returning."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with p.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def provenance(kind):
    paths = [PROTOCOL, *REPO.glob('tools/m8i1a_*.py'),
             *REPO.glob('tests/test_m8i1a*.py')]
    return dict(kind=kind, base_commit=BASE,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                source_sha256={str(p.relative_to(REPO)): sha(p) for p in sorted(paths)},
                actor_creations=0, sensor_creations=0, TM_instances=0, owner_ticks=0,
                model_forwards=0, training_steps=0, visual_control=False, new_Final=0)


def config():
    return json.loads(PROTOCOL.read_text())
