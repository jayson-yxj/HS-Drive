"""Shared M5-J read-only data selection and strict artifact helpers."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
ROOT='/root/gpufree-data/hs-drive-data/processed/m5i_pixels'
TEST='/root/gpufree-data/hs-drive-data/processed/m5j_test_pixels'
def rows(split):
    root=TEST if split=='test'else '/root/gpufree-data/hs-drive-data/processed/m5i_test_pixels'if split=='historical'else ROOT
    return [r for r in json.loads(Path(root,'index.json').read_text())['records']if split=='historical'or r['split']==split]
def write(path,value):
    if Path(path).exists():raise FileExistsError(path)
    atomic_json(path,value)
def source():return {str(p):file_hash(p)for root in ['tools','models','tests','configs','diagnostics']for p in Path(root).glob('*m5j*')if p.is_file()}
def quantile(x):
    x=np.asarray(x);return dict(n=x.size,mean=float(x.mean())if x.size else None,quantiles=np.quantile(x,[.1,.5,.9,.99]).tolist()if x.size else [])
