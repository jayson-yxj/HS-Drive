"""Read-only historical assurance and complete stage artifact inventory."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    expected=json.loads(Path('reports/m6f_execution_logs/base_freeze.json').read_text())['files']
    changed=[name for name,h in expected.items() if not Path(name).is_file() or digest(Path(name))!=h]
    if changed:raise RuntimeError('Historical changes: '+str(changed))
    files=[]
    for root in ['reports','docs','tools','simulator','tests','configs/m6f']:
        for p in Path(root).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p) and ('m6f' in str(p).lower()) and p.name!='m6f_release_manifest.json':
                files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    out=dict(kind='actual_server_append_only_release_inventory',base='b0ee249',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             historical_files_verified=len(expected),files=sorted(files,key=lambda x:x['path']),
             self_reference='Manifest excludes itself; final commit contains this manifest; execution commits remain in batch records',
             local_external_assets=[dict(path='/root/gpufree-data/simulators/m6f_CARLA_0.9.16.tar.gz',sha256='09e3ebb28df17962f0c997e66f4b914ad5ea6f1d6a6dbbf13c9f87eb38346d57')])
    with Path('reports/m6f_release_manifest.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(len(files),'M6F assets;',len(expected),'historical files unchanged')

if __name__=='__main__':main()
