"""Rehash historical tracked files and frozen historical assets, read-only."""
import json
from pathlib import Path
from tools.m6b_common import write,provenance,asset
from simulator.recorder import file_hash


def main():
    freeze=json.loads(Path('runs/m6b_logs/freeze.json').read_text());failures=[];counts={}
    for kind in ['tracked','assets']:
        count=0;size=0
        for name,sha in freeze[kind].items():
            p=Path(name)
            if not p.exists():failures.append(dict(path=name,reason='missing'));continue
            current=file_hash(p)
            if current!=sha:failures.append(dict(path=name,reason='SHA_changed',original=sha,current=current))
            size+=p.stat().st_size;count+=1
        counts[kind]=dict(checked=count,bytes=size)
        print(kind,count,'failures',len(failures),flush=True)
    original=freeze['m6a_index'];actual=asset(original['path'])
    if original!=actual:failures.append(dict(path=original['path'],reason='m6a_index_changed'))
    write('reports/m6b_preservation.json',dict(**provenance('actual_server_SHA256_verification'),
        freeze=asset('runs/m6b_logs/freeze.json'),counts=counts,failures=failures,passed=not failures,
        limits='Verifies all base tracked files and inherited 50867 frozen historical assets; not a new inventory of every unrelated server file.'))

if __name__=='__main__':main()
