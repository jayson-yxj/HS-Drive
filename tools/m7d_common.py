"""M7-D append-only provenance; no inherited assets modified."""
from pathlib import Path
import subprocess
from tools.m7a_common import read,write,sha,asset
BASE='d9899fac94f320195f85bb6fe34d7c52752c58d7'
def provenance():
 files=[p for root in ['tools','datasets','diagnostics','tests','configs/m7d'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m7d' in p.name or root=='configs/m7d')]
 return dict(base=BASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in files},new_planner_control=False,new_training=False)
