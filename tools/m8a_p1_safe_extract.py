"""CPU-only weights-only pickle reader, inside restricted no-network filesystem."""
import hashlib,json,os
from pathlib import Path
import torch
import numpy as np
assert os.environ.get('M8A_ISOLATED')=='1'
state=torch.load('/input/epoch_24.pth',map_location='cpu',weights_only=True)['state_dict']
arrays={key:value.numpy() for key,value in state.items()}
assert len(arrays)==558 and all(np.isfinite(a).all() for a in arrays.values())
np.savez('/output/model_tensors.npz',**arrays)
p=Path('/output/model_tensors.npz')
Path('/output/safe_extract.json').write_text(json.dumps(dict(status='Passed',method='torch.load_weights_only_CPU_restricted_bwrap',tensor_count=len(arrays),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),output_format='NPZ_numeric_arrays_only_allow_pickle_False',optimizer_exported=False),indent=2))
