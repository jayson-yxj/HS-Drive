"""Verify old assets and seal new source/data/checkpoints/reports without modifying them."""
import json,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_artifact_manifest.json')
    if dest.exists():raise FileExistsError(dest)
    baseline=json.loads(Path('reports/m5f_baseline_freeze.json').read_text());old={}
    for category in ['tracked_sha256','checkpoint_sha256']:
        bad=[p for p,sha in baseline[category].items() if not Path(p).exists() or file_hash(p)!=sha];old[category]=dict(checked=len(baseline[category]),mismatches=bad)
        if bad:raise RuntimeError(f'Oldasset changed: {bad}')
    roots=[*Path('runs').glob('m5f*'),Path('/root/gpufree-data/hs-drive-data/raw/m5f_sealed'),Path('/root/gpufree-data/hs-drive-data/processed/m5f_pixels'),Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels'),Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_index')];assets={}
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:assets[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    reused={}
    for root in ['m5f_pixels','m5f_test_pixels']:
        q=Path('/root/gpufree-data/hs-drive-data/processed',root,'quality.json');m=json.loads(q.read_text())
        for p,sha in m['source_assets_sha256'].items():
            if file_hash(p)!=sha:raise RuntimeError(f'Rawsource changed: {p}')
            reused[p]=sha
    sources={str(p):file_hash(p) for d in ['tools','models','datasets','diagnostics','training','tests','configs','docs'] for p in Path(d).glob('*m5f*') if p.is_file()};sources['docs/M5F_IMPLEMENTATION.md']=file_hash('docs/M5F_IMPLEMENTATION.md');reports={str(p):file_hash(p) for p in Path('reports').rglob('*') if p.is_file() and (p.name.startswith('m5f_') or any(x.startswith('m5f_')for x in p.parts)) and p!=dest};atomic_json(dest,dict(kind='filesystem_SHA256_provenance_and_preservation_check',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),old_assets=old,sources=sources,assets=assets,reused_raw_sources=reused,reports=reports,limits='Manifest excludes itself to avoid selfhash. Git commit identified before report commit. Oldtracked4071files plusold78checkpoints untouched; reusedrawbytesverified againstinitialadapterhash. Externaldata/models notembeddedinGit.'))

if __name__=='__main__':main()
