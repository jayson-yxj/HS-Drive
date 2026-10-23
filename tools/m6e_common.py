"""Append-only M6-E artifacts and source provenance."""
import json,subprocess
from pathlib import Path
from simulator.recorder import file_hash
BASE='ba5ba6e759b807d5e3740061bebf72b7d1ced565'

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def asset(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p))

def provenance(kind):
    files=[]
    for d in ['simulator','tools','tests','configs/m6e']:
        files.extend(p for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6e' in p.name.lower() or d=='configs/m6e'))
    return dict(kind=kind,source='Privileged',base=BASE,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):file_hash(p) for p in sorted(files)},visual_control=False)
