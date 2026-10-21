"""Historical integrity and new versioned assets, no overwrite of old experiments."""
import json,subprocess
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5j_common import write,source

def main():
    old=json.loads(Path('reports/m5j_baseline_freeze.json').read_text());checks={}
    for k in ['tracked','assets']:
        bad=[p for p,h in old[k].items()if file_hash(p)!=h];assert not bad,bad;checks[k]=dict(checked=len(old[k]),mismatches=bad)
    assets={}
    for root in [*Path('runs').glob('m5j*'),*Path('/root/gpufree-data/hs-drive-data/raw').glob('m5j*'),*Path('/root/gpufree-data/hs-drive-data/processed').glob('m5j*')]:
        for p in root.rglob('*'):
            if p.is_file():assets[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    reports={str(p):file_hash(p)for p in Path('reports').rglob('*')if p.is_file()and any(k.startswith('m5j_')for k in p.parts)}
    write('reports/m5j_artifact_manifest.json',dict(kind='actual_filesystem_full_integrity',historical=checks,new_assets=assets,sources=source(),reports=reports,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),limits='Large raw/cache retainedserver notGit. Originalmodels reused read-only no newtraincheckpoint.'))
if __name__=='__main__':main()
