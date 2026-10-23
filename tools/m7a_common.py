"""Stage-local immutable artifacts and provenance."""
import hashlib,json,subprocess
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def asset(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def provenance():
    sources=[p for root in ['tools','datasets','diagnostics','tests','configs/m7a'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m7a' in p.name or root=='configs/m7a')]
    return dict(stage='M7-A',base='eb2c8f409632cfcdfe610b538b2634c540558232',
      commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      sources={str(p):sha(p) for p in sources},protocol=asset('configs/m7a/protocol.json'),
      default_modules_modified=False,visual_control=False,new_training_steps=0)
