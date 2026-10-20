"""Load the installed CUDA linalg library by absolute path on this container."""
import ctypes
from pathlib import Path
import torch

_HANDLE=None


def prepare_cuda():
    global _HANDLE
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    path=Path(torch.__file__).parent/'lib/libtorch_cuda_linalg.so'
    if path.exists() and _HANDLE is None:
        _HANDLE=ctypes.CDLL(str(path))
    return str(path)
