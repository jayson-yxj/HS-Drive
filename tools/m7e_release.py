"""Read-only historic preservation and append-only release manifest."""
import argparse
from pathlib import Path
from tools.m7e_common import *
from tools.m7a_release import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    if not a.seal:
        checks=dict(base_tracked=verify(read('reports/m7e_execution_logs/base_freeze.json')['files']),M7D_manifest=verify({x['path']:x['sha256'] for x in read('reports/m7d_release_manifest.json')['files']}),inherited_external=verify(read('runs/m6b_logs/freeze.json')['assets']),CARLA0916_binary=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']))
        write('reports/m7e_execution_logs/preservation.json',dict(kind='ActualSHA256Verification',status='Passed',checks=checks,changed=[]));print(checks)
    else:
        files=[asset(p) for root in ['reports','tools','datasets','diagnostics','tests','docs','configs/m7e'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'm7e' in str(p).lower() and p.name!='m7e_release_manifest.json']
        write('reports/m7e_release_manifest.json',dict(kind='ActualServerAssetSeal',provenance=provenance(),files=files,preservation=read('reports/m7e_execution_logs/preservation.json'),self_reference='Manifest excluded; protected by final Git commit; execution sources individually hashed.'));print('sealed',len(files))
if __name__=='__main__':main()
