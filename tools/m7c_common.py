"""M7-C append-only artifacts and byte provenance."""
from pathlib import Path
import subprocess
from tools.m7a_common import read,write,sha,asset
BASE='684b6e64bfd7fce89be48a0a9599ee22c8c25226'
def provenance():
 files=[p for root in ['tools','diagnostics','tests','configs/m7c'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m7c' in p.name or root=='configs/m7c')]
 return dict(base=BASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):sha(p) for p in files},execution='OfflineCARLA0915Map_SyntheticActors',new_control=False,new_training=False)


def equal(a,b,path=''):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k],path+'/'+k) for k,v in a.items() if k!='latency_ms')
    if isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b):return False
        if a and a[0]=='latency_ms':return True
        return all(equal(x,y,path) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and abs(a-b)<=1e-9
    return a==b
