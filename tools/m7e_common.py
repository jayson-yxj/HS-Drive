"""Append-only M7E assets and source provenance."""
from pathlib import Path
import subprocess
from tools.m7a_common import read,write,sha,asset
BASE='6ed646ccffbafe760f5bba8d01b58be94ed11089'
def provenance():
    files=[p for root in ['tools','datasets','diagnostics','tests','configs/m7e'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m7e' in str(p).lower())]
    return dict(base=BASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in files},new_control=False,new_training=False)
