"""Read-only preservation plus stage SHA seal."""
import argparse,subprocess
from pathlib import Path
from tools.m7a_common import read,write,sha,asset


def verify(mapping):
    changed=[p for p,h in mapping.items() if not Path(p).is_file() or sha(p)!=h]
    if changed:raise RuntimeError(str(changed))
    return len(mapping)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');a=parser.parse_args()
    baseline=read('reports/m7a_execution_logs/base_freeze.json')
    checks=dict(base_tracked_files=verify(baseline['files']),
      inherited_external_assets=verify(read('runs/m6b_logs/freeze.json')['assets']),
      M6G_assets=verify({x['path']:x['sha256'] for x in read('reports/m6g_release_manifest.json')['files']}),
      CARLA0916_binary_files=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']))
    if a.seal:
        files=[]
        for root in ['reports','tools','datasets','diagnostics','tests','docs','configs/m7a']:
            files.extend(asset(p) for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p)
              and 'm7a' in str(p).lower() and p.name!='m7a_release_manifest.json')
        write('reports/m7a_release_manifest.json',dict(kind='ActualServerAppendOnlyAssetInventory',
          artifact_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          files=sorted(files,key=lambda v:v['path']),checks=checks,
          self_reference='manifest excluded; final commit protects manifest; individual execution SHA retained'))
    else:
        write('reports/m7a_execution_logs/preservation.json',dict(kind='ActualServerSHA256Verification',status='Passed',checks=checks,
          changed=[],scope='Manifested bytes only',base=baseline['base']))
    print(checks)

if __name__=='__main__':main()
