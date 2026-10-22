"""Freeze original tracked tree and historical experiment assets without mutation."""
import json, subprocess
from pathlib import Path
from tools.m5o_common import BASE, DATA, write, asset, provenance
from simulator.recorder import file_hash


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == BASE
    tracked = {p:file_hash(p) for p in subprocess.check_output(['git','ls-files'],text=True).splitlines()}
    original = json.loads((DATA/'m5n_asset_sha_index.json').read_text())['assets']
    inherited = json.loads(Path('runs/m5n_logs/baseline.json').read_text())['assets']
    paths = set(original) | set(inherited)
    paths.update(str(p) for p in Path('runs').glob('m5n*/**/*') if p.is_file())
    assets = {p:file_hash(p) for p in sorted(paths) if Path(p).is_file()}
    differences = [p for p,v in original.items() if p in assets and assets[p]!=v['sha256']]
    write('runs/m5o_logs/historical_freeze.json',dict(base=BASE,tracked=tracked,assets=assets,prior_index_differences=differences))
    write('reports/m5o_baseline_freeze.json',dict(**provenance(),historical_tracked=len(tracked),historical_assets=len(assets),prior_index_differences=differences,index=asset('runs/m5o_logs/historical_freeze.json')))
    print('Frozen',len(tracked),len(assets),differences)

if __name__=='__main__':main()
