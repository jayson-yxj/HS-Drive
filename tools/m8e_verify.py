"""Read-only reproducibility verification; no CARLA/model imports."""
import json
from pathlib import Path
from tools.m8a_common import sha


def verify(path):
    data=json.loads(Path(path).read_text())
    failures=[]
    for a in data['files']:
        p=Path(a['path'])
        if not p.is_file() or sha(p)!=a['sha256']:
            failures.append(a['path'])
    if failures:
        raise AssertionError(dict(manifest=str(path),failures=failures))
    return len(data['files'])


def main():
    for name in ['m8b_release_manifest','m8b_evaluation_reference_audit',
                 'm8c_release_manifest','m8d_release_manifest','m8e_release_manifest']:
        print(name,verify(Path('reports')/(name+'.json')),'SHA verified',flush=True)


if __name__=='__main__':main()
