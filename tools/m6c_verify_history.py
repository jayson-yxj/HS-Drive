"""Read-only preservation check against M6-B base plus inherited asset hashes."""
import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m6c_common import write,provenance,asset


def main():
    freeze=json.loads(Path('runs/m6c_logs/freeze.json').read_text());old=json.loads(Path('runs/m6b_logs/freeze.json').read_text());counts={};errors=[]
    for kind,items in [('base_tracked',freeze['tracked']),('historical_assets',old['assets'])]:
        total=0
        for name,sha in items.items():
            p=Path(name)
            if not p.exists() or file_hash(p)!=sha:errors.append(name)
            else:total+=1
        counts[kind]=total;print(kind,total,flush=True)
    if Path('reports/m6c_preservation.json').exists():
        previous=json.loads(Path('reports/m6c_preservation.json').read_text())
        assert previous['counts']==counts and previous['errors']==errors
        print('Repeated SHA verification agrees with preserved report',flush=True)
        return
    write('reports/m6c_preservation.json',dict(**provenance('actual_server_history_SHA256_revalidation'),passed=not errors,counts=counts,errors=errors,freeze=asset('runs/m6c_logs/freeze.json')))

if __name__=='__main__':main()
