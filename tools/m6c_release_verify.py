"""Seal all deliverables with exact source/config/assets SHA; no old mutations."""
import json,subprocess
from pathlib import Path
from tools.m6c_common import write,asset,provenance


def main():
    required=['summary.md','acceptance.json','low_speed_dynamics.json','startup_modes.json','coast_brake_envelope.json','goal_geometry_regression.json','stop_abcde_comparison.json','no_bypass_safety.json','runtime_tail.json','safety_regression.json']
    for name in required:
        p=Path('reports')/('m6c_'+name);assert p.is_file()
        if p.suffix=='.json':json.loads(p.read_text())
    a=json.loads(Path('reports/m6c_acceptance.json').read_text());assert a['status']=='Partial' and not a['default_replacement'] and not a['visual_control']
    assert a['stop']['by_arm']['D']['goal_passed']==0 and a['E_policy_search']=='NotRun'
    assert a['preservation']['passed']
    assert all(q['failures']==q['errors']==0 for q in a['tests'])
    files=[]
    for p in Path('reports').glob('m6c_*'):
        if p.name=='m6c_release_manifest.json':continue
        files.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    for directory in ['tools','simulator','tests','docs','configs/m6c']:
        files.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6c' in p.name.lower() or directory=='configs/m6c'))
    write('reports/m6c_release_manifest.json',dict(**provenance('actual_server_release_SHA256_seal'),passed=True,files=[asset(p) for p in sorted(set(files))]))
    print('sealed',len(set(files)))

if __name__=='__main__':main()
