"""Immutable old assets plus complete new research hashes; no model redistribution."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    p=Path('reports/m5i_artifact_manifest.json')
    if p.exists():raise FileExistsError(p)
    old=json.loads(Path('reports/m5i_baseline_freeze.json').read_text());checks={}
    for key in ['tracked_sha256','assets_sha256']:
        bad=[f for f,h in old[key].items()if not Path(f).exists()or file_hash(f)!=h];checks[key]=dict(checked=len(old[key]),mismatches=bad)
        if bad:raise RuntimeError(bad)
    assets={}
    for root in [*Path('runs').glob('m5i*'),*Path('/root/gpufree-data/hs-drive-data/raw').glob('m5i*'),*Path('/root/gpufree-data/hs-drive-data/processed').glob('m5i*')]:
        for f in root.rglob('*'):
            if f.is_file():assets[str(f)]=dict(sha256=file_hash(f),bytes=f.stat().st_size)
    sources={str(f):file_hash(f)for root in ['tools','models','training','tests','configs','diagnostics']for f in Path(root).glob('*m5i*')if f.is_file()};reports={str(f):file_hash(f)for f in Path('reports').rglob('*')if f.is_file()and any(v.startswith('m5i_')for v in f.parts)and f!=p};atomic_json(p,dict(kind='actual_filesystem_integrity_and_provenance',preservation=checks,assets=assets,sources=sources,reports=reports,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),limits='Large raw/features/checkpoints retainedserver notGit; noexternalweightredistribution. Manifestexcludesitself.'))
if __name__=='__main__':main()
