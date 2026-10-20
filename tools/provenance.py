"""Source/config fingerprint when a handoff workspace has no Git history yet."""
import hashlib
from pathlib import Path


def source_hashes(root='.'):
    root=Path(root)
    paths=[]
    for folder in ('geometry','simulator','datasets','tools','configs','models','training'):
        paths.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.py','.yaml'))
    for name in ('requirements-sim.lock.txt','requirements-train.lock.txt','pyproject.toml'):
        path=root/name
        if path.exists():
            paths.append(path)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
