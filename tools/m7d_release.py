"""Preserve historical bytes and seal all M7D artifacts; never rewrite old files."""
import argparse,subprocess
from pathlib import Path
from tools.m7d_common import *
from tools.m7a_release import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    if not a.seal:
        base=read('reports/m7d_execution_logs/base_freeze.json')
        checks=dict(base_tracked_files=verify(base['files']),inherited_external_assets=verify(read('runs/m6b_logs/freeze.json')['assets']),
                    M7C_assets=verify({x['path']:x['sha256'] for x in read('reports/m7c_release_manifest.json')['files']}),
                    CARLA0916_binary_files=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']))
        write('reports/m7d_execution_logs/preservation.json',dict(kind='ActualServerSHA256Verification',status='Passed',checks=checks,changed=[],base=BASE))
        print(checks)
    else:
        files=[]
        for root in ['reports','tools','datasets','diagnostics','tests','docs','configs/m7d']:
            for f in Path(root).rglob('*'):
                if f.is_file() and '__pycache__' not in str(f) and 'm7d' in str(f).lower() and f.name!='m7d_release_manifest.json':files.append(asset(f))
        write('reports/m7d_release_manifest.json',dict(kind='ActualServerAppendOnlyAssetManifest',provenance=provenance(),files=sorted(files,key=lambda v:v['path']),
            preservation=read('reports/m7d_execution_logs/preservation.json'),self_reference='Manifest excluded; final Git commit protects manifest; each execution has its own source SHA.'))
        print('sealed',len(files),'files')
if __name__=='__main__':main()
