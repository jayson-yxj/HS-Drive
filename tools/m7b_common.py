"""Append-only M7-B artifact and source provenance."""
from pathlib import Path
import subprocess
from tools.m7a_common import read, write, sha, asset
BASE='3d149ac872d2b8e17b3e6010bc90ecdb555d2bf4'
def provenance():
    files=[p for root in ['tools','diagnostics','tests','configs/m7b'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m7b' in p.name or root=='configs/m7b')]
    return dict(base=BASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in files},execution='OfflineCARLA0915Map_SyntheticActors',new_control=False,new_training=False)
