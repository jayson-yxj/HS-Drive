"""Read-only historical SHA assurance."""
import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m6e_common import write,provenance

def main():
    frozen=json.loads(Path('reports/m6e_execution_logs/base_freeze.json').read_text());assets=json.loads(Path('runs/m6b_logs/freeze.json').read_text())['assets'];counts={};errors=[]
    for kind,files in [('base_tracked',frozen['files']),('inherited_assets',assets)]:
        good=0
        for name,sha in files.items():
            p=Path(name)
            if p.is_file() and file_hash(p)==sha:good+=1
            else:errors.append(name)
        counts[kind]=good;print(kind,good,flush=True)
    write('reports/m6e_preservation.json',dict(**provenance('actual_server_historical_SHA_verification'),counts=counts,errors=errors,passed=not errors))
if __name__=='__main__':main()
