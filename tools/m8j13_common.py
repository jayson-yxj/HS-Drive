"""J13 provenance and exclusive-output helpers; private payloads stay private."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def config():return json.loads(Path('configs/m8j13_protocol.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,allow_nan=False)
    os.replace(tmp,path)
def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
def sources():
    return {str(p):digest(p) for pat in ('tools/m8j13*.py','simulator/m8j13*.py','tests/m8j13/*.py','configs/m8j13*.json') for p in Path('.').glob(pat)}
def provenance():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources(),protocol_sha256=digest('configs/m8j13_protocol.json'))
def quantiles(values):
    import numpy as np
    if not values:return None
    return {k:float(v) for k,v in zip(('p50','p95','p99','max'),np.percentile(values,[50,95,99,100]))}


def assets():
    c=config();root=Path(c['runtime_root']);old=json.loads(Path(c['asset_protocol']).read_text())
    commit=subprocess.check_output(['git','-C',str(root/'official-source'),'rev-parse','HEAD'],text=True).strip()
    assert commit==old['official_commit']
    files=[]
    for row in json.loads(Path('reports/m8j12_sources/asset_finish.json').read_text()):
        p=root/'artifacts'/row['file'];h=digest(p)
        assert h==row['sha256'] and p.stat().st_size==row['actual_bytes']
        files.append(dict(file=row['file'],bytes=p.stat().st_size,sha256=h))
    for rel,h in old['source_hashes'].items():assert digest(root/'official-source'/rel)==h
    runtime_differences=[]
    for p in (root/'official-source').rglob('*.py'):
        rel=p.relative_to(root/'official-source');other=root/'runtime-source'/rel
        if not other.exists() or digest(p)!=digest(other):runtime_differences.append(str(rel))
    assert not runtime_differences,runtime_differences
    for row in json.loads(Path('reports/m8j12_sources/runtime_source_equivalence.json').read_text())['compiled_extensions']:
        assert digest(row['path'])==row['sha256']
    return dict(official_commit=commit,files=files,official_source_hashes=old['source_hashes'],runtime_python_differences=runtime_differences,extensions_unchanged=True)
