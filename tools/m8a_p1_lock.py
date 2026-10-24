"""Append-only runtime/wheel/extension identity and original asset preservation."""
import json
from pathlib import Path
import subprocess
from tools.m8a_p1_common import ROOT,LOG,CFG,write,asset,sha,provenance,environment

def main():
    code='import json,sys,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":sorted([[d.metadata["Name"],d.version] for d in m.distributions()])}))'
    packages=json.loads(subprocess.check_output([str(ROOT/'env/bin/python'),'-c',code],env=environment(),text=True))
    wheels=[asset(p) for p in sorted((ROOT/'wheels').iterdir()) if p.is_file()]
    binaries=[asset(p) for p in sorted((ROOT/'sources/mmdetection3d/mmdet3d/ops').rglob('*.so'))]
    binaries.append(asset(ROOT/'env/lib/python3.8/site-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so'))
    sources=[asset(p) for p in sorted((ROOT/'sources/mmdetection3d').rglob('*')) if p.is_file() and 'build' not in p.parts and p.suffix in ['.py','.cpp','.cu','.h','.cc','.txt']]
    frozen=[asset(p) for p in sorted((ROOT/'sources/PETR/projects').rglob('*.py'))]
    protected=[]
    for name in ['reports/m8a_proxy_retry_manifest.json','reports/m8a_release_manifest.json']:
        manifest=json.loads(Path(name).read_text())
        for entry in manifest['files']:
            path=Path(entry['path']);protected.append(dict(path=str(path),unchanged=path.exists() and sha(path)==entry['sha256']))
    assert all(x['unchanged'] for x in protected)
    changed=subprocess.check_output(['git','diff','--name-status',CFG['base']],text=True).splitlines()
    assert all(row.startswith('A\t') and row.split('\t')[-1].startswith(('tools/m8a_p1','tests/test_m8a_p1','configs/m8a/p1_','reports/m8a_p1','docs/M8A_P1')) for row in changed),changed
    (LOG/'pip_freeze.txt').write_text(subprocess.check_output([str(ROOT/'env/bin/python'),'-m','pip','freeze','--all'],env=environment(),text=True))
    write('reports/m8a_p1_environment_lock.json',dict(**provenance('ActualIsolatedEnvironmentIdentity'),root=str(ROOT),packages=packages,wheels_and_sources=wheels,extensions=binaries,
        mmdet3d_source_files=sources,petr_source_files=frozen,checkpoint=asset(CFG['checkpoint']),
        main_environments= json.loads((LOG/'main_environment_preservation.json').read_text()),historical_assets=protected,
        compatibility_patch_logs=[asset(p) for p in sorted(LOG.glob('*.patch'))],pip_freeze=asset(LOG/'pip_freeze.txt')))

if __name__=='__main__':main()
