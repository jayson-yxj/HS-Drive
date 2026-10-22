"""Verify append-only M6-B release evidence and seal every artifact with SHA256."""
import json,subprocess
from pathlib import Path
from tools.m6b_common import asset,provenance,write
from simulator.recorder import file_hash


def main():
    required=['acceptance','goal_stop_geometry','longitudinal_calibration','low_speed_regression','bypass_feasibility','oracle_maneuver_results','runtime_tail','safety_regression','runtime_full_campaign','tests','preservation']
    for name in required:json.loads(Path(f'reports/m6b_{name}.json').read_text())
    a=json.loads(Path('reports/m6b_acceptance.json').read_text())
    assert a['status']=='Partial' and not a['visual_control']
    assert a['trial_counts']['completed']==69 and a['trial_counts']['initialization_failures']==2
    assert a['low_speed']['by_arm']['C']['goal_passed']==6
    assert a['static_bypass']['refusal_actual_OBB_gap_3m_passed']==1
    assert a['normal_navigation']['strict_passed']==0
    assert a['safety']['expired_normal_control']==0
    assert a['runtime']['all69_trials_max_ms']>170
    assert a['preservation']['passed'] and a['tests']['passed']
    paths=[]
    for p in sorted(Path('reports').glob('m6b_*')):
        if p.name=='m6b_release_manifest.json':continue
        paths.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    for directory in ['tools','simulator','tests','docs','configs/m6b']:
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6b' in p.name.lower() or directory=='configs/m6b'))
    report=dict(**provenance('actual_server_release_consistency_and_SHA256_seal'),passed=True,
        artifacts=[asset(p) for p in sorted(set(paths))],
        earlier_manifest='m6b_asset_manifest.json is the initial seal; summary and final diagnostics are finalized in this release seal.',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write('reports/m6b_release_manifest.json',report)
    print('release consistency passed',len(report['artifacts']),'files')

if __name__=='__main__':main()
