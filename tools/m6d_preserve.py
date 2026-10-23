"""Verify immutable base and inherited asset SHA; no mutations to old reports."""
import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m6d_common import write,provenance,asset

def main():
    base=json.loads(Path('reports/m6d_execution_logs/base_freeze.json').read_text());historical=json.loads(Path('runs/m6b_logs/freeze.json').read_text())
    counts={};errors=[]
    for kind,files in [('M6C_tracked',base['tracked']),('inherited_assets',historical['assets'])]:
        good=0
        for name,sha in files.items():
            p=Path(name)
            if not p.is_file() or file_hash(p)!=sha:errors.append(name)
            else:good+=1
        counts[kind]=good;print(kind,good,flush=True)
    write('reports/m6d_preservation.json',dict(**provenance('actual_server_history_SHA_verification'),passed=not errors,counts=counts,errors=errors,freeze=asset('reports/m6d_execution_logs/base_freeze.json')))
if __name__=='__main__':main()
