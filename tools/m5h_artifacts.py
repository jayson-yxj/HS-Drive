"""Preserve historical assets and hash every new study asset and experiment source."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5h_artifact_manifest.json')
    if out.exists():raise FileExistsError(out)
    baseline=json.loads(Path('reports/m5h_baseline_freeze.json').read_text());checks={}
    for key in ['tracked_sha256','assets_sha256']:
        bad=[p for p,h in baseline[key].items()if not Path(p).exists()or file_hash(p)!=h];checks[key]=dict(checked=len(baseline[key]),mismatches=bad)
        if bad:raise RuntimeError('Priorassetschanged '+str(bad))
    assets={}
    for root in [*Path('runs').glob('m5h*'),*Path('/root/gpufree-data/hs-drive-data/raw').glob('m5h*'),*Path('/root/gpufree-data/hs-drive-data/processed').glob('m5h*')]:
        for p in root.rglob('*'):
            if p.is_file():assets[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    sources={str(p):file_hash(p)for root in ['tools','training','diagnostics','tests','configs']for p in Path(root).glob('*m5h*')if p.is_file()};reports={str(p):file_hash(p)for p in Path('reports').rglob('*')if p.is_file()and any(part.startswith('m5h_')for part in p.parts)and p!=out};atomic_json(out,dict(kind='actual_filesystem_SHA_audit',preservation=checks,assets=assets,sources=sources,reports=reports,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),limits='Largecheckpoint/rawdata serverassets notGit; teacherweights notrepublished. Manifestexcludesitself.'))
if __name__=='__main__':main()
