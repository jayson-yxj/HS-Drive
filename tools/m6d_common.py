"""Append-only M6-D provenance and independently named experiment outputs."""
import json,subprocess
from pathlib import Path
from simulator.recorder import file_hash

BASE='d808c21c1a4a2a1e7a2a14c5a4a1dc5465768fc1'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def asset(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p))

def provenance(kind):
    files=[]
    for d in ['simulator','tools','tests','configs/m6d']:
        files.extend(p for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6d' in p.name.lower() or d=='configs/m6d'))
    return dict(kind=kind,source='Privileged',base=BASE,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p):file_hash(p) for p in sorted(files)},visual_control=False)
