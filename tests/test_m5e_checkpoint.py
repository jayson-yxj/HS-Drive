import numpy as np
import torch
from tools.m5e_common import load_checkpoint


def test_restricted_checkpoint_numpy_selection_metadata(tmp_path):
    path=tmp_path/'ck.pt'
    torch.save({'weights':torch.ones(2),'selection':np.float64(.3)},path)
    ck=load_checkpoint(path)
    assert ck['selection']==.3 and torch.equal(ck['weights'],torch.ones(2))
