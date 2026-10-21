"""Compact delivery index, full hashes stay in an independent server asset."""
import json,subprocess
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5k_common import write,asset

def main():
 old=json.loads(Path('runs/m5k_logs/baseline_files.json').read_text());checks={}
 for key in ['tracked','assets']:
  bad=[p for p,h in old[key].items()if file_hash(p)!=h];assert not bad,bad;checks[key]=dict(checked=len(old[key]),mismatches=bad)
 assets={}
 for root in [*Path('runs').glob('m5k*'),*Path('/root/gpufree-data/hs-drive-data/raw').glob('m5k*'),*Path('/root/gpufree-data/hs-drive-data/processed').glob('m5k*')]:
  for p in root.rglob('*'):
   if p.is_file():assets[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
 path=Path('/root/gpufree-data/hs-drive-data/m5k_asset_sha_index.json');write(path,dict(assets=assets,sources={str(p):file_hash(p)for root in ['tools','models','simulator','training','configs','tests']for p in Path(root).glob('*m5k*')if p.is_file()}));write('reports/m5k_artifact_manifest.json',dict(kind='actual_filesystem_integrity',historical=checks,index=asset(path),new_asset_count=len(assets),new_bytes=sum(v['bytes']for v in assets.values()),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),limits='Large raw/cache/log records remainserver; SHAindex referenced. Historical Git/assets verified unchanged.'))
if __name__=='__main__':main()
