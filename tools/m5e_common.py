"""Checked M5-E supervision readers; never passed to learned forward inputs."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import file_hash

ROOT='/root/gpufree-data/hs-drive-data/processed/m5e_index'


def load_checkpoint(path):
    """Restricted loader for our tensors and NumPy float64 selection metadata."""
    import torch
    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))
    ]):
        return torch.load(path, weights_only=True)


def checked_npz(path,sha):
    if file_hash(path)!=sha:raise ValueError('Checksum mismatch')
    with np.load(path) as z:return {k:z[k] for k in z.files}


def sidecars(index,rows,name):
    records=json.loads(Path(index,name).read_text())['records'];lookup={r['sample_id']:r for r in records}
    return [checked_npz(lookup[r['sample_id']]['path'],lookup[r['sample_id']]['sha256']) for r in rows]


def pairs(index,split):
    rows=json.loads(Path(index,'m5e_pairs.json').read_text())['pairs']
    return [(r,checked_npz(r['path'],r['sha256'])) for r in rows if r['split']==split]
