"""Preserve baseline and hash all study data/checkpoints/sources/results."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5g_artifact_manifest.json')
    if dest.exists():raise FileExistsError(dest)
    old=json.loads(Path('reports/m5g_baseline_freeze.json').read_text());checks={}
    for key in ['tracked_sha256','checkpoint_sha256']:
        bad=[p for p,h in old[key].items()if not Path(p).exists()or file_hash(p)!=h];checks[key]=dict(checked=len(old[key]),mismatches=bad)
        if bad:raise RuntimeError(bad)
    olddata=json.loads(Path('reports/m5f_artifact_manifest.json').read_text());bad=[]
    for p,h in olddata['reused_raw_sources'].items():
        if file_hash(p)!=h:bad.append(p)
    checks['historical_raw_sources']=dict(checked=len(olddata['reused_raw_sources']),mismatches=bad)
    if bad:raise RuntimeError(bad)
    assets={};roots=[*Path('runs').glob('m5g*'),Path('/root/gpufree-data/hs-drive-data/raw/m5g_sealed'),Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_index'),Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_pixels')]
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file()and '__pycache__'not in p.parts:assets[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    sources={str(p):file_hash(p)for d in ['tools','training','tests','configs']for p in Path(d).glob('*m5g*')if p.is_file()};sources['docs/M5G_IMPLEMENTATION.md']=file_hash('docs/M5G_IMPLEMENTATION.md');reports={str(p):file_hash(p)for p in Path('reports').rglob('*')if p.is_file()and any(x.startswith('m5g_')for x in p.parts)and p!=dest};atomic_json(dest,dict(kind='actual_filesystem_SHA_provenance',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),preservation=checks,sources=sources,assets=assets,reports=reports,weight=json.loads(Path('reports/m5g_pretrained_provenance.json').read_text()),limits='Manifestexcludesitself. Raw/modeldata retainedserver notGit. Sourcecommit beforefinalreportcommit; nooldfilechanges.'))

if __name__=='__main__':main()
