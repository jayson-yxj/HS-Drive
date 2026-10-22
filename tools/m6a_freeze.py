"""Verify and freeze historical M4--M5-O assets; never modify them."""
import json
import subprocess
from pathlib import Path
from simulator.recorder import file_hash
from tools.m6a_common import BASE, asset, provenance, write


def main():
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], text=True).splitlines()
    hashes = {p: file_hash(p) for p in tracked}
    index_path = Path('/root/gpufree-data/hs-drive-data/m5o_asset_sha_index.json')
    prior = json.loads(index_path.read_text())['assets']
    missing, changed, verified = [], [], {}
    for p, ref in prior.items():
        if not Path(p).is_file():
            missing.append(p)
            continue
        sha = file_hash(p)
        verified[p] = sha
        if sha != ref['sha256']:
            changed.append(p)
    write('runs/m6a_logs/historical_freeze.json', dict(base=BASE, tracked=hashes, assets=verified))
    write('reports/m6a_historical_integrity.json', dict(
        **provenance('actual_server_SHA256_verification'),
        tracked_count=len(hashes), asset_count=len(prior), verified=len(verified),
        missing=missing, changed=changed, status='Passed' if not missing and not changed else 'Failed',
        original_index=asset(index_path), freeze=asset('runs/m6a_logs/historical_freeze.json')))
    print(len(hashes), len(verified), 'missing', len(missing), 'changed', len(changed), flush=True)


if __name__ == '__main__':
    main()
