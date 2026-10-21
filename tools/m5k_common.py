"""Versioned M5-K artifacts; large records live outside Git."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import file_hash
from tools.m5j_common import write
ROOT=Path('/root/gpufree-data/hs-drive-data/processed')

def rows(split):
    if split=='historical_val':
        from tools.m5j_common import rows as old
        return old('val')
    root=ROOT/('m5k_test_pixels' if split=='test' else 'm5k_dev_pixels')
    return [r for r in json.loads((root/'index.json').read_text())['records'] if r['split']==split]

def asset(path):
    return dict(path=str(path),sha256=file_hash(path),bytes=Path(path).stat().st_size)

def summary(values):
    x=np.array(values);return dict(n=len(x),mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max())) if len(x) else dict(n=0)

def group_depth(records,keys):
    groups={}
    for r in records:
        key=tuple(r[k] for k in keys); a=groups.setdefault(key,dict(n=0,sums={}))
        a['n']+=r['n']
        for k,v in r['sums'].items():a['sums'][k]=a['sums'].get(k,0)+v
    return [dict(zip(keys,key),n=a['n'],**{k:v/a['n'] for k,v in a['sums'].items()})for key,a in groups.items() if a['n']]
