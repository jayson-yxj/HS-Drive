"""Early byte integrity check; does not finalize ongoing new experiments."""
import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5o_common import write,provenance

def main():
    baseline=json.loads(Path('runs/m5o_logs/historical_freeze.json').read_text());bad=[]
    for kind in ['tracked','assets']:
        for p,h in baseline[kind].items():
            if not Path(p).is_file() or file_hash(p)!=h:bad.append([kind,p])
    write('reports/m5o_historical_integrity_pretraining.json',dict(kind='actual_filesystem_SHA256_audit',changed=bad,tracked=len(baseline['tracked']),assets=len(baseline['assets']),provenance=provenance()))
    assert not bad,bad
    print('Historical bytes unchanged',len(baseline['tracked']),len(baseline['assets']))
if __name__=='__main__':main()
