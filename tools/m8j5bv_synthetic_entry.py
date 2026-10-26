"""Run core synthetic suite with private datasets explicitly denied."""
import os
from pathlib import Path
import sys


def main():
    denied=('m8j3-bench2drive','m8j4-single-frame','m8j5-supervision','m8j5a-raw-audit','m8j5b-coverage-supervision')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            parts=Path(os.fsdecode(args[0])).absolute().parts
            if any(part.startswith(name) for part in parts for name in denied):
                raise PermissionError('CoreSyntheticPrivateDataForbidden')
    sys.addaudithook(guard)
    for name in (*denied,'m8j3-bench2drive-single-clip','m8j4-single-frame-gt'):
        try:Path('/root/gpufree-data/hs-drive-private/'+name+'/canary').read_bytes()
        except PermissionError:pass
        else:raise AssertionError('Private denial failed')
    import pytest
    raise SystemExit(pytest.main(['--noconftest','tests/test_m8j5bv_loss.py','-q','-ra']))


if __name__=='__main__':main()
