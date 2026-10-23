"""Verify required M6-D delivery and seal source/config/raw evidence SHA."""
import json,re,subprocess
from pathlib import Path
from tools.m6d_common import write,asset,provenance

def main():
    names=['summary.md','acceptance.json','signed_motion_contract.json','goal_obstacle_target_separation.json','hold_release_feasibility.json','startup_restart_modes.json','controller_ablation.json','rollback_regression.json','stop_safety_comparison.json','comfort_feasibility.json','runtime_status.json']
    for name in names:
        p=Path('reports')/('m6d_'+name);assert p.is_file(),str(p)
        if p.suffix=='.json':json.loads(p.read_text())
    a=json.loads(Path('reports/m6d_acceptance.json').read_text())
    assert a['decisions']['complete_signed_motion_protection'].startswith('NoGo')
    assert a['decisions']['F1'].startswith('NotRun') and a['decisions']['sealed_regression'].startswith('NotRun')
    assert a['preservation']['passed'] and not a['visual_control']
    assert all(q['failures']==q['errors']==0 and q['xfailed']==1 for q in a['tests'])
    for directory,count in [('m6d_probe_artifacts',38),('m6d_trials',28)]:
        assert len(list(Path('reports',directory).glob('*/result.json')))==count
    for ref in re.findall(r'\]\((m6d_figures/[^)]+)\)',Path('reports/m6d_summary.md').read_text()):assert (Path('reports')/ref).is_file()
    changes=subprocess.check_output(['git','diff','--name-status','d808c21'],text=True).splitlines()
    assert all(s.startswith('A\t') for s in changes),changes
    files=[]
    for p in Path('reports').glob('m6d_*'):
        if p.name=='m6d_release_manifest.json':continue
        files.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    for directory in ['tools','simulator','tests','docs','configs/m6d']:
        files.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6d' in p.name.lower() or directory=='configs/m6d'))
    write('reports/m6d_release_manifest.json',dict(**provenance('actual_server_release_SHA256_seal'),passed=True,
        files=[asset(p) for p in sorted(set(files))],limits='Artifact integrity only; does not upgrade failed motion, comfort or runtime gates.'))
    print('sealed',len(set(files)),'files')
if __name__=='__main__':main()
