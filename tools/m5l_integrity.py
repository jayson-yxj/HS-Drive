"""Preserve historical identities; external SHA index for large artifacts."""
import json,subprocess
from pathlib import Path
from tools.m5l_common import write,asset
from simulator.recorder import file_hash

def main():
 old=json.loads(Path('runs/m5l_logs/baseline.json').read_text());checks={}
 for key in ['tracked','assets']:
  bad=[p for p,h in old[key].items()if file_hash(p)!=h];assert not bad,bad;checks[key]=dict(checked=len(old[key]),mismatches=bad)
 freeze=json.loads(Path('reports/m5l_pretest_freeze.json').read_text());bad=[p for p,h in freeze['sources'].items()if file_hash(p)!=h];assert not bad,bad
 files={}
 for root in [*Path('runs').glob('m5l*'),*Path('/root/gpufree-data/hs-drive-data/raw').glob('m5l*'),*Path('/root/gpufree-data/hs-drive-data/processed').glob('m5l*')]:
  for p in root.rglob('*'):
   if p.is_file():files[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
 path=Path('/root/gpufree-data/hs-drive-data/m5l_asset_sha_index.json');write(path,dict(assets=files,sources={str(p):file_hash(p)for root in ['tools','models','diagnostics','simulator','configs','tests']for p in Path(root).glob('*m5l*')if p.is_file()}));write('reports/m5l_artifact_manifest.json',dict(kind='actual_filesystem_fullhistoricalverification',historical=checks,scientific_sources_unchanged=True,index=asset(path),count=len(files),bytes=sum(v['bytes']for v in files.values()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
if __name__=='__main__':main()
